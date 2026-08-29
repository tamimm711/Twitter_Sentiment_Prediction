import pandas as pd

from sklearn.model_selection import cross_validate, GridSearchCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


def universal_nlp_classifier(
    X_train,
    y_train,

    X_val=None,
    y_val=None,

    X_test=None,
    y_test=None,

    cv=5,
    scoring="f1_weighted"
):

    # ========================================
    # 1. Define NLP Models
    # ========================================

    models = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=2000,
                random_state=42
            ),

        "Multinomial Naive Bayes":
            MultinomialNB(),

        "Linear SVC":
            LinearSVC(
                random_state=42
            ),

        "SGD Classifier":
            SGDClassifier(
                random_state=42
            )
    }


    # ========================================
    # 2. Compare Models Using Cross Validation
    # ========================================

    results = []

    for name, model in models.items():

        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,

            scoring={
                "accuracy": "accuracy",
                "precision": "precision_weighted",
                "recall": "recall_weighted",
                "f1": "f1_weighted"
            },

            n_jobs=-1
        )

        results.append({

            "Model": name,

            "CV Accuracy":
                round(scores["test_accuracy"].mean(), 4),

            "CV Precision":
                round(scores["test_precision"].mean(), 4),

            "CV Recall":
                round(scores["test_recall"].mean(), 4),

            "CV F1":
                round(scores["test_f1"].mean(), 4)
        })


    # ========================================
    # 3. Create Results DataFrame
    # ========================================

    results_df = pd.DataFrame(results)

    results_df = (
        results_df
        .sort_values(
            by="CV F1",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print("\nModel Comparison:")
    print(results_df)


    # ========================================
    # 4. Select Best Model
    # ========================================

    best_model_name = results_df.iloc[0]["Model"]

    best_base_model = models[best_model_name]

    print(
        "\nBest Model Based on CV F1:",
        best_model_name
    )


    # ========================================
    # 5. Hyperparameter Grids
    # ========================================

    param_grids = {

        "Logistic Regression": {
            "C": [0.01, 0.1, 1, 10, 100],
            "class_weight": [None, "balanced"]
        },

        "Multinomial Naive Bayes": {
            "alpha": [0.01, 0.1, 0.5, 1.0, 2.0]
        },

        "Linear SVC": {
            "C": [0.01, 0.1, 1, 10],
            "class_weight": [None, "balanced"]
        },

        "SGD Classifier": {
            "loss": ["hinge", "log_loss"],
            "alpha": [0.00001, 0.0001, 0.001],
            "penalty": ["l2", "l1", "elasticnet"]
        }
    }


    # ========================================
    # 6. Tune Best Model
    # ========================================

    grid_search = GridSearchCV(
        estimator=best_base_model,
        param_grid=param_grids[best_model_name],
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    grid_search.fit(
        X_train,
        y_train
    )

    best_model = grid_search.best_estimator_

    print("\nBest Parameters:")
    print(grid_search.best_params_)

    print("\nBest CV Score After Tuning:")
    print(
        round(
            grid_search.best_score_,
            4
        )
    )


    # ========================================
    # 7. Validation Evaluation
    # ========================================

    val_pred = None

    if (
        X_val is not None
        and
        y_val is not None
    ):

        val_pred = best_model.predict(X_val)

        print("\nValidation Performance:")

        print(
            "Accuracy :",
            round(
                accuracy_score(y_val, val_pred),
                4
            )
        )

        print(
            "Precision:",
            round(
                precision_score(
                    y_val,
                    val_pred,
                    average="weighted",
                    zero_division=0
                ),
                4
            )
        )

        print(
            "Recall   :",
            round(
                recall_score(
                    y_val,
                    val_pred,
                    average="weighted",
                    zero_division=0
                ),
                4
            )
        )

        print(
            "F1 Score :",
            round(
                f1_score(
                    y_val,
                    val_pred,
                    average="weighted",
                    zero_division=0
                ),
                4
            )
        )

        print("\nValidation Confusion Matrix:")

        print(
            confusion_matrix(
                y_val,
                val_pred
            )
        )

        print("\nValidation Classification Report:")

        print(
            classification_report(
                y_val,
                val_pred,
                zero_division=0
            )
        )


    # ========================================
    # 8. Test Evaluation
    # ========================================

    test_pred = None

    if (
        X_test is not None
        and
        y_test is not None
    ):

        test_pred = best_model.predict(X_test)

        print("\nTest Performance:")

        print(
            "Accuracy :",
            round(
                accuracy_score(
                    y_test,
                    test_pred
                ),
                4
            )
        )

        print(
            "Precision:",
            round(
                precision_score(
                    y_test,
                    test_pred,
                    average="weighted",
                    zero_division=0
                ),
                4
            )
        )

        print(
            "Recall   :",
            round(
                recall_score(
                    y_test,
                    test_pred,
                    average="weighted",
                    zero_division=0
                ),
                4
            )
        )

        print(
            "F1 Score :",
            round(
                f1_score(
                    y_test,
                    test_pred,
                    average="weighted",
                    zero_division=0
                ),
                4
            )
        )

        print("\nTest Confusion Matrix:")

        print(
            confusion_matrix(
                y_test,
                test_pred
            )
        )

        print("\nTest Classification Report:")

        print(
            classification_report(
                y_test,
                test_pred,
                zero_division=0
            )
        )


    # ========================================
    # 9. Return Results
    # ========================================

    return {

        "model": best_model,

        "results": results_df,

        "best_model_name": best_model_name,

        "best_params": grid_search.best_params_,

        "best_cv_score": grid_search.best_score_,

        "val_pred": val_pred,

        "test_pred": test_pred
    }