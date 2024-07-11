import torch, os, random, time, namegenerator, datetime, configparser
import torchvision.models as models
import torchvision.datasets as datasets
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from data.uerc_dataset import UERCDataset

# Set up the run ID and other necessary parameters:
random.seed(torch.seed())  # random for RUN_ID
RUN_ID = (
    datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S") + " " + namegenerator.gen()
)
print("\nRun ID:", RUN_ID + "")
# Fixed seeds for reproducibility:
torch.manual_seed(42)
random.seed(42)

os.chdir(os.path.dirname(os.path.abspath(__file__)))
config = configparser.ConfigParser()
config.read("config.ini")

# For more info on the parameters, see the config.ini file
TRAIN_CLASSES_COUNT = config.getint("BASELINE_MODEL", "number_of_classes_in_training")
BASELINE_WEIGHTS = config.get("BASELINE_MODEL", "weights")

PLOT_FORMAT = "." + config.get("OUTPUT_SETTINGS", "plot_format")
STORE_FEATURE_VECTORS = config.getboolean("OUTPUT_SETTINGS", "store_features")
DRAW_AND_STORE_PLOTS = config.getboolean("OUTPUT_SETTINGS", "store_plots")
STORE_RESULTS = config.getboolean("OUTPUT_SETTINGS", "store_results")

RUN_PATH = os.path.join(config.get("OUTPUT_PATHS", "runs_dir"), RUN_ID)
FEATURES_PATH = os.path.join(RUN_PATH, config.get("OUTPUT_PATHS", "features_dir"))
PLOTS_PATH = os.path.join(RUN_PATH, config.get("OUTPUT_PATHS", "plots"))
RESULTS_CSV = os.path.join(RUN_PATH, config.get("OUTPUT_PATHS", "results_csv"))


class UERC23:
    device = None
    uerc_dataset = None

    def __init__(
        self, images_path, annotations_csv, full_image_list_csv, data_splits_csv=None
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.images_path = images_path
        self.annotations_csv = annotations_csv

        os.makedirs("runs", exist_ok=True)
        self.uerc_dataset = UERCDataset(
            "test",
            self.images_path,
            self.annotations_csv,
            full_image_list_csv,
            data_splits_csv,
        )

        if STORE_RESULTS:
            os.makedirs(RUN_PATH, exist_ok=True)

        print("Running on:", str(self.device).upper())
        print(
            "Storing feature vectors:", FEATURES_PATH if STORE_FEATURE_VECTORS else "No"
        )
        print(
            "Drawing plots:",
            (
                os.path.join(PLOTS_PATH, "[covariate]" + PLOT_FORMAT)
                if DRAW_AND_STORE_PLOTS
                else "No"
            ),
        )
        print("Storing results:", RESULTS_CSV if STORE_RESULTS else "No")
        print("")

    def log(self, s):
        print(s, end="")
        if STORE_RESULTS:
            with open(RESULTS_CSV, "a+") as f:
                f.write(s)

    def gini_coefficient(self, eer_values):
        n = len(eer_values)
        eer_sum = np.sum(eer_values)
        mean_eer = eer_sum / n

        sum_abs_diff = 0
        for i in range(n):
            for j in range(n):
                sum_abs_diff += abs(eer_values[i] - eer_values[j])

        gini = sum_abs_diff / (2 * n * n * mean_eer)
        return gini * 100

    def weighted_average_rank(self, scores, ginis, score_weight=0.8, gini_weight=0.2):
        # Higher the score, better the rank
        # Lower the gini, better the rank
        rank = score_weight * np.mean(scores) + gini_weight * np.mean(ginis)
        return rank

    def load_feature_vectors(self, features_path, covariate_name):
        # Load feature vectors from CSV file:
        f = os.path.join(features_path, covariate_name + ".csv")
        features = np.genfromtxt(f, delimiter=" ")

        # Load class labels from CSV file:
        f = os.path.join(features_path, covariate_name + ".classes.csv")
        labels = np.genfromtxt(f, dtype=int)

        return features, labels

    def compute_feature_vectors(self, folder_list, covariate_name):

        # Load the pre-trained ResNet model
        model_path = os.path.join(BASELINE_WEIGHTS)
        model = models.resnet18()
        # Replace the last fully connected layer:
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, TRAIN_CLASSES_COUNT)
        model.eval()
        model.load_state_dict(torch.load(model_path), strict=False)

        # Get the feature extractor and set it to eval mode:
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-1]).eval()
        # Move the model to the device:
        feature_extractor.to(self.device)

        # Load the dataset
        dataset = datasets.ImageFolder(
            self.images_path, transform=self.uerc_dataset.transforms["test"]
        )

        # Filter the dataset to only include the desired subfolders:
        dataset.samples = [
            s for s in dataset.samples if s[0].split("/")[-2] in folder_list
        ]
        dataset.targets = [s[1] for s in dataset.samples]

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1024, shuffle=False, num_workers=0
        )

        feature_vectors = torch.tensor([]).to(self.device)
        classes = torch.tensor([])
        with torch.no_grad():
            for i, (image, class_) in enumerate(dataloader):
                feature_vector = feature_extractor(image.to(self.device))
                feature_vectors = torch.cat((feature_vectors, feature_vector), dim=0)

                classes = torch.cat((classes, class_), dim=0)
        torch.cuda.empty_cache()

        feature_vectors = feature_vectors.cpu().numpy()
        feature_vectors = feature_vectors.reshape(feature_vectors.shape[0], -1)

        if STORE_FEATURE_VECTORS:
            os.makedirs(FEATURES_PATH, exist_ok=True)

            # Save the feature vectors to a txt file
            np.savetxt(
                os.path.join(FEATURES_PATH, covariate_name + ".csv"),
                feature_vectors,
                delimiter=" ",
            )

            # Save classes to a txt file
            np.savetxt(
                os.path.join(FEATURES_PATH, covariate_name + ".classes.csv"),
                classes,
                delimiter=" ",
                fmt="%d",
            )

        return feature_vectors, classes

    def compute_performance(self, features, labels, covariate_name):

        # Compute cosine similarity between each pair of feature vectors
        cosine_sim = cosine_similarity(features)

        # Set the diagonal of cosine_sim to the worse possible value
        np.fill_diagonal(cosine_sim, -1)

        # Compute ROC curve based on the cosine similarity

        labels = np.array(labels)
        cosine_sim = np.array(cosine_sim)
        uniq = np.unique(labels)

        # Prepare new matrix CC which contains the average values for each class (element of the labels array):
        CC = np.zeros((len(uniq), cosine_sim.shape[1]))
        for i, class_ in enumerate(uniq):
            CC[i] = np.max(cosine_sim[labels == class_], axis=0)

        fpr = {}
        tpr = {}
        roc_auc = {}

        for i, class_ in enumerate(uniq):
            labels_binary = labels == class_
            fpr[class_], tpr[class_], _ = roc_curve(labels_binary, CC[i])
            roc_auc[class_] = roc_auc_score(labels_binary, CC[i])

        n_classes = len(np.unique(labels))
        n_points = max([len(fpr[class_]) for class_ in np.unique(labels)])
        fpr_mean = np.zeros(n_points)
        tpr_mean = np.zeros(n_points)

        for class_ in np.unique(labels):
            fpr_interp = np.interp(
                np.arange(n_points), np.arange(len(fpr[class_])), fpr[class_]
            )
            tpr_interp = np.interp(
                np.arange(n_points), np.arange(len(tpr[class_])), tpr[class_]
            )
            fpr_mean += fpr_interp
            tpr_mean += tpr_interp

        fpr_mean /= n_classes
        tpr_mean /= n_classes

        fpr = fpr_mean
        tpr = tpr_mean

        roc_auc = sk.metrics.auc(fpr, tpr)

        # find the threshold that gives equal false positive rate (FPR) and false negative rate (FNR)
        fnr = 1 - tpr
        min_index = np.argmin(
            np.abs(fpr - fnr)
        )  # same as np.argmin(np.sqrt((fpr ** 2) + ((1 - tpr) ** 2)))
        eer = fpr[min_index]

        # Compute VER@1%FAR: Verification Error Rate at 1% False Acceptance Rate or False Rejection Rate (FRR or FNMR or Verification Error Rate - VER) at 1% False Acceptance Rate (FAR)
        min_index = np.searchsorted(fpr, 0.01, side="left")
        frr = 1 - tpr[min_index]
        ver = frr

        if DRAW_AND_STORE_PLOTS:
            os.makedirs(PLOTS_PATH, exist_ok=True)

            # Plot ROC curve
            plt.figure()
            lw = 2
            plt.plot(
                fpr,
                tpr,
                color="darkorange",
                lw=lw,
                label="ROC curve (area = %0.2f)" % roc_auc,
            )
            plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
            # Also draw the line going from the top left corner to the bottom right corner, make a line thinner:
            plt.plot([0, 1], [1, 0], color="gray", lw=0.5, linestyle="--")
            plt.plot(
                [eer], [1 - eer], marker="o", markersize=5, label="EER = %0.2f" % eer
            )
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Receiver operating characteristic curve")
            plt.legend(loc="lower right")

            # plt.show()
            # Store plot in a PNG file:
            plt.savefig(os.path.join(PLOTS_PATH, covariate_name + PLOT_FORMAT), dpi=300)
            plt.close()

        # Compute rank-1 accuracy
        # Find the index of the maximum similarity for each feature vector
        pred_labels = labels[cosine_sim.argmax(axis=1)]
        r1 = np.mean(pred_labels == labels)

        return roc_auc * 100, eer * 100, ver * 100, r1 * 100

    def compute_results_for_one_covariate(
        self, folder_list, covariate_name, precomputed_features_path
    ):

        if precomputed_features_path is None:
            # We either need to compute feature vectors by extracting features:
            features, labels = self.compute_feature_vectors(folder_list, covariate_name)
        else:
            # Or we can load them from CSV file:
            features, labels = self.load_feature_vectors(
                precomputed_features_path, covariate_name
            )

        # Now we can compute similarity matrix, and compute all the scores:
        roc_auc, eer, ver, r1 = self.compute_performance(
            features, labels, covariate_name
        )

        img_count = features.shape[0]
        class_count = np.unique(labels).shape[0]

        return img_count, class_count, roc_auc, eer, ver, r1

    def compute_results_for_all(self, precomputed_features_path):
        start_time = time.time()

        # This automatically selects 10% of images for evaluation. However, if you are using precomputed features this is used only for covariate iteration (but actual images are  not used):
        # selected_images = self.prepare_eval_data()
        selected_images = self.uerc_dataset.subjects_covariate_test

        i = 0
        scores = np.zeros((4, len(selected_images)))
        s = "Covariate\tImg#\tCla#\tAUC[%]\tEER[%]\tV1F[%]\tR1[%]\n"
        self.log(s)
        for covariate, subject_list in selected_images.items():
            covariate_form = " - ".join(covariate).title()
            img_count, class_count, roc_auc, eer, ver, r1 = (
                self.compute_results_for_one_covariate(
                    subject_list, covariate_form, precomputed_features_path
                )
            )
            scores[:, i] = roc_auc, eer, ver, r1

            s = covariate_form + "\t{}\t{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n".format(
                img_count, class_count, roc_auc, eer, ver, r1
            )
            self.log(s)
            i += 1

        mean_auc = np.mean(scores[0])
        mean_eer = np.mean(scores[1])
        mean_ver = np.mean(scores[2])
        gini_eer = self.gini_coefficient(scores[1])

        track1 = self.weighted_average_rank(mean_eer, gini_eer)

        print("")
        s = "{:.2f}% UERC Ranking (lower is beter)\n".format(track1)
        self.log(s)

        print(
            "Means: AUC ({:.2f}%), EER ({:.2f}%), VER@1%FAR ({:.2f}%)\n".format(
                mean_auc, mean_eer, mean_ver
            )
        )

        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print(
            f"Elapsed time: {minutes}m {seconds}s.\n"
            if minutes > 0
            else f"Elapsed time: {seconds}s.\n"
        )


if __name__ == "__main__":

    # Folder of the UERC23 dataset:
    images_path = os.path.join("data", "public")

    # File that contains gender and ethnicity annotations:
    annotations_csv = os.path.join("data", "public_annotations.csv")

    # OPTIONAL, meant for speedup. If the file does not exist, it will be created:
    full_image_csv = os.path.join("runs", "public_image_list.csv")

    # OPTIONAL, meant for speedup. If the file does not exist, it will be created. Also  feel free to split data your own way. To do that provide your own CSV file with the same format as the one generated here:
    data_splits_csv = os.path.join("runs", "train_val_test_splits.csv")  # optional,

    uerc23 = UERC23(
        images_path, annotations_csv, full_image_csv, data_splits_csv=data_splits_csv
    )

    # You can either load your own feature vectors when you use your own prediction model. Note that for each demographic group there needs to be a separate [x].csv file and [x].classes.csv. To see how this looks like:
    # - either set precomputed_features_path to None (as set below), the code will generate feature vectors in runs/ folder
    # - or take a look at "precomputed_feature_vectors_example" folder, which contains feature vectors for the baseline model. If set here, no feature extraction is performed, only the performance is computed:
    precomputed_features_path = "precomputed_feature_vectors_example"

    # Or you can compute feature vectors here, using the baseline model, by setting precomputed_features_path to None. By default 10% of images are used for evaluation, but feel free to change that:
    precomputed_features_path = None

    # Then we can compute the performance:
    uerc23.compute_results_for_all(precomputed_features_path)
