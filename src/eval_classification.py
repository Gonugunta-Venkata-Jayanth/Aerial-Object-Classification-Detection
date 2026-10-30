import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from config import CLASS_MODELS_DIR, REPORTS_DIR
from data_classification import get_datasets

if __name__ == "__main__":
    # Choose which model to evaluate
    model_path = CLASS_MODELS_DIR / "best_transfer_model.h5"
    # model_path = CLASS_MODELS_DIR / "best_cnn_model.h5"

    _, _, test_ds, class_names = get_datasets()
    print("Evaluating model:", model_path)

    model = tf.keras.models.load_model(model_path)

    y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
    y_pred_prob = model.predict(test_ds)
    y_pred = (y_pred_prob > 0.5).astype("int32").reshape(-1)

    # classification report
    report_str = classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
    print(report_str)

    with open(REPORTS_DIR / "classification_report.txt", "w") as f:
        # store a repo-relative path so the report is portable across machines
        f.write("Model: " + model_path.relative_to(CLASS_MODELS_DIR.parents[1]).as_posix() + "\n\n")
        f.write(report_str)

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png")
    plt.close()

    print("Evaluation complete. Reports saved in:", REPORTS_DIR)
