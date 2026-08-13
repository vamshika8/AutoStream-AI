from flask import Flask, render_template, request
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    r2_score,
    mean_absolute_error,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

app = Flask(__name__)

uploaded_data = None

trained_model = None
model_features = None
model_problem_type = None
target_encoder = None

feature_importance = None
confusion_matrix_data = None
confusion_labels = None

model_accuracy = None
model_precision = None
model_recall = None
model_f1 = None

best_model_name = None
model_comparison = []


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():
    return render_template("index.html")


# =========================================================
# UPLOAD CSV
# =========================================================

@app.route("/upload", methods=["POST"])
def upload():

    global uploaded_data

    file = request.files.get("file")

    if file is None or file.filename == "":
        return "Please select a CSV file."

    if not file.filename.lower().endswith(".csv"):

        return """
        <h2>⚠️ Invalid File</h2>
        <p>Please upload a CSV file only.</p>
        <a href="/">Go Back</a>
        """

    try:

        data = pd.read_csv(file)

        if data.empty:

            return """
            <h2>⚠️ Empty Dataset</h2>
            <p>The uploaded CSV does not contain any rows.</p>
            <a href="/">Go Back</a>
            """

        rows = len(data)
        columns = len(data.columns)

        if columns < 2:

            return """
            <h2>⚠️ Dataset Not Suitable</h2>

            <p>
            AutoStream needs at least two columns:
            features and a target column.
            </p>

            <a href="/">Go Back</a>
            """

        column_names = list(data.columns)

        ignored_columns = []

        for column in data.columns:

            name = column.lower().strip()

            if name in [
                "id",
                "index",
                "serial_number",
                "serial number"
            ]:

                ignored_columns.append(column)

        analysis_data = data.drop(
            columns=ignored_columns,
            errors="ignore"
        )

        analysis_data = analysis_data.dropna(
            axis=1,
            how="all"
        )

        missing_before = int(
            analysis_data.isnull().sum().sum()
        )

        numeric_columns = list(
            analysis_data.select_dtypes(
                include="number"
            ).columns
        )

        categorical_columns = list(
            analysis_data.select_dtypes(
                include="object"
            ).columns
        )

        # Numeric missing values
        for column in numeric_columns:

            if analysis_data[column].isnull().sum() > 0:

                median_value = (
                    analysis_data[column].median()
                )

                analysis_data[column] = (
                    analysis_data[column]
                    .fillna(median_value)
                )

        # Categorical missing values
        for column in categorical_columns:

            if analysis_data[column].isnull().sum() > 0:

                mode_value = (
                    analysis_data[column].mode()
                )

                if not mode_value.empty:

                    analysis_data[column] = (
                        analysis_data[column]
                        .fillna(mode_value[0])
                    )

        missing_after = int(
            analysis_data.isnull().sum().sum()
        )

        cleaned_values = (
            missing_before - missing_after
        )

        uploaded_data = analysis_data.copy()

        statistics = []

        numeric_data = analysis_data.select_dtypes(
            include="number"
        )

        for column in numeric_data.columns:

            statistics.append({

                "feature": column,

                "mean": round(
                    numeric_data[column].mean(),
                    2
                ),

                "median": round(
                    numeric_data[column].median(),
                    2
                ),

                "minimum": round(
                    numeric_data[column].min(),
                    2
                ),

                "maximum": round(
                    numeric_data[column].max(),
                    2
                ),

                "std": round(
                    numeric_data[column].std(),
                    2
                ),

                "unique": int(
                    numeric_data[column].nunique()
                )

            })

        correlation = {}

        if len(numeric_data.columns) > 1:

            correlation = (
                numeric_data
                .corr()
                .round(2)
                .to_dict()
            )

        return render_template(
            "result.html",

            filename=file.filename,

            rows=rows,

            columns=columns,

            column_names=column_names,

            ignored_columns=ignored_columns,

            numeric_columns=numeric_columns,

            categorical_columns=categorical_columns,

            total_missing=missing_before,

            missing_after=missing_after,

            cleaned_values=cleaned_values,

            statistics=statistics,

            correlation=correlation
        )

    except Exception as e:

        return f"""
        <h2>⚠️ Could Not Read CSV</h2>

        <p>{e}</p>

        <a href="/">Go Back</a>
        """


# =========================================================
# TRAIN MODEL
# =========================================================

@app.route("/train", methods=["POST"])
def train():

    global uploaded_data
    global trained_model
    global model_features
    global model_problem_type
    global target_encoder

    global feature_importance
    global confusion_matrix_data
    global confusion_labels

    global model_accuracy
    global model_precision
    global model_recall
    global model_f1

    global best_model_name
    global model_comparison

    target = request.form.get("target")

    if uploaded_data is None:

        return """
        <h2>⚠️ No Dataset</h2>

        <p>Please upload a CSV file first.</p>

        <a href="/">Go Back</a>
        """

    if not target:

        return """
        <h2>⚠️ Target Column Missing</h2>

        <p>Please select a target column.</p>

        <a href="/">Go Back</a>
        """

    try:

        data = uploaded_data.copy()

        if target not in data.columns:

            return """
            <h2>⚠️ Invalid Target</h2>

            <p>
            The selected target column was not found.
            </p>

            <a href="/">Go Back</a>
            """

        data = data.dropna(
            subset=[target]
        )

        if len(data) < 10:

            return """
            <h2>⚠️ Not Enough Data</h2>

            <p>
            AutoStream needs at least 10 usable rows
            to train a model.
            </p>

            <a href="/">Go Back</a>
            """

        X = data.drop(
            columns=[target]
        )

        y = data[target]

        unique_target = y.nunique()

        if unique_target < 2:

            return """
            <h2>⚠️ Target Column Not Suitable</h2>

            <p>
            The selected target contains only one
            unique value.
            </p>

            <a href="/">Go Back</a>
            """

        if unique_target >= len(y) * 0.9:

            return """
            <h2>⚠️ Target Column Not Suitable</h2>

            <p>
            The selected target looks like an ID,
            name, or unique identifier.
            </p>

            <p>
            Please select a meaningful outcome column.
            </p>

            <a href="/">Go Back</a>
            """

        # =================================================
        # DETECT PROBLEM TYPE
        # =================================================

        if y.dtype == "object":

            problem_type = "Classification"

        elif y.nunique() <= 10:

            problem_type = "Classification"

        else:

            problem_type = "Regression"

        # =================================================
        # PREPARE FEATURES
        # =================================================

        X = pd.get_dummies(
            X,
            drop_first=True
        )

        X = X.apply(
            pd.to_numeric,
            errors="coerce"
        )

        X = X.fillna(0)

        X = X.loc[
            :,
            X.nunique() > 1
        ]

        if X.shape[1] == 0:

            return """
            <h2>⚠️ Dataset Not Suitable</h2>

            <p>
            AutoStream could not find usable feature
            columns for machine learning.
            </p>

            <p>
            Please upload a structured dataset with
            meaningful feature columns.
            </p>

            <a href="/">Go Back</a>
            """

        model_features = list(
            X.columns
        )

        # =================================================
        # CLASSIFICATION
        # =================================================

        if problem_type == "Classification":

            target_encoder = None

            if y.dtype == "object":

                target_encoder = LabelEncoder()

                y = target_encoder.fit_transform(y)

                confusion_labels = list(
                    target_encoder.classes_
                )

            else:

                confusion_labels = [
                    str(x)
                    for x in sorted(
                        y.unique()
                    )
                ]

            # Make sure stratification is possible
            class_counts = pd.Series(y).value_counts()

            if class_counts.min() < 2:

                return """
                <h2>⚠️ Not Enough Samples</h2>

                <p>
                Each class needs at least two samples
                for model training.
                </p>

                <a href="/">Go Back</a>
                """

            X_train, X_test, y_train, y_test = (
                train_test_split(
                    X,
                    y,
                    test_size=0.2,
                    random_state=42,
                    stratify=y
                )
            )

            # =================================================
            # THREE MODELS
            # =================================================

            models = {

                "Random Forest": RandomForestClassifier(
                    n_estimators=100,
                    random_state=42
                ),

                "Logistic Regression": Pipeline([
                    (
                        "scaler",
                        StandardScaler()
                    ),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=1000,
                            random_state=42
                        )
                    )
                ]),

                "Decision Tree": DecisionTreeClassifier(
                    random_state=42
                )

            }

            comparison = []

            trained_models = {}

            # Train and compare
            for name, current_model in models.items():

                try:

                    current_model.fit(
                        X_train,
                        y_train
                    )

                    current_predictions = (
                        current_model.predict(
                            X_test
                        )
                    )

                    current_accuracy = (
                        accuracy_score(
                            y_test,
                            current_predictions
                        ) * 100
                    )

                    current_precision = (
                        precision_score(
                            y_test,
                            current_predictions,
                            average="weighted",
                            zero_division=0
                        ) * 100
                    )

                    current_recall = (
                        recall_score(
                            y_test,
                            current_predictions,
                            average="weighted",
                            zero_division=0
                        ) * 100
                    )

                    current_f1 = (
                        f1_score(
                            y_test,
                            current_predictions,
                            average="weighted",
                            zero_division=0
                        ) * 100
                    )

                    comparison.append({

                        "model": name,

                        "accuracy": round(
                            current_accuracy,
                            2
                        ),

                        "precision": round(
                            current_precision,
                            2
                        ),

                        "recall": round(
                            current_recall,
                            2
                        ),

                        "f1": round(
                            current_f1,
                            2
                        )

                    })

                    trained_models[name] = (
                        current_model
                    )

                except Exception:

                    continue

            if len(comparison) == 0:

                return """
                <h2>⚠️ Model Training Failed</h2>

                <p>
                None of the available models could
                be trained on this dataset.
                </p>

                <a href="/">Go Back</a>
                """

            # =================================================
            # SELECT BEST MODEL
            # =================================================

            comparison.sort(
                key=lambda x:
                    x["f1"],
                reverse=True
            )

            best_model_name = (
                comparison[0]["model"]
            )

            trained_model = trained_models[
                best_model_name
            ]

            # Best model metrics
            best_result = comparison[0]

            model_accuracy = (
                best_result["accuracy"]
            )

            model_precision = (
                best_result["precision"]
            )

            model_recall = (
                best_result["recall"]
            )

            model_f1 = (
                best_result["f1"]
            )

            model_comparison = comparison

            model_problem_type = (
                "Classification"
            )

            # Predictions using best model
            predictions = trained_model.predict(
                X_test
            )

            confusion_matrix_data = (
                confusion_matrix(
                    y_test,
                    predictions
                ).tolist()
            )

            # Feature importance
            if hasattr(
                trained_model,
                "feature_importances_"
            ):

                importances = (
                    trained_model
                    .feature_importances_
                )

            else:

                # Logistic Regression
                if hasattr(
                    trained_model,
                    "named_steps"
                ):

                    coefficients = (
                        trained_model
                        .named_steps["model"]
                        .coef_
                    )

                    importances = (
                        abs(
                            coefficients
                        ).mean(axis=0)
                    )

                else:

                    importances = [
                        0
                        for _ in model_features
                    ]

        # =================================================
        # REGRESSION
        # =================================================

        else:

            target_encoder = None

            y = pd.to_numeric(
                y,
                errors="coerce"
            )

            valid = y.notna()

            X = X.loc[valid]

            y = y.loc[valid]

            if len(X) < 10:

                return """
                <h2>⚠️ Not Enough Numeric Data</h2>

                <p>
                The selected target does not contain
                enough usable numeric values.
                </p>

                <a href="/">Go Back</a>
                """

            X_train, X_test, y_train, y_test = (
                train_test_split(
                    X,
                    y,
                    test_size=0.2,
                    random_state=42
                )
            )

            trained_model = RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )

            trained_model.fit(
                X_train,
                y_train
            )

            predictions = trained_model.predict(
                X_test
            )

            r2 = r2_score(
                y_test,
                predictions
            )

            mae = mean_absolute_error(
                y_test,
                predictions
            )

            model_accuracy = round(
                r2 * 100,
                2
            )

            model_precision = None
            model_recall = None
            model_f1 = None

            model_problem_type = (
                "Regression"
            )

            best_model_name = (
                "Random Forest Regressor"
            )

            model_comparison = []

            confusion_matrix_data = None
            confusion_labels = None

            importances = (
                trained_model
                .feature_importances_
            )

            mae_value = round(
                mae,
                2
            )

        # =================================================
        # FEATURE IMPORTANCE
        # =================================================

        feature_importance = []

        for feature, importance in zip(
            model_features,
            importances
        ):

            feature_importance.append({

                "feature": feature,

                "importance": round(
                    float(importance) * 100,
                    2
                )

            })

        feature_importance.sort(
            key=lambda x:
                x["importance"],
            reverse=True
        )

        # =================================================
        # SAVE MODEL
        # =================================================

        model_package = {

            "model": trained_model,

            "features": model_features,

            "problem_type":
                model_problem_type,

            "target_encoder":
                target_encoder,

            "accuracy":
                model_accuracy,

            "precision":
                model_precision,

            "recall":
                model_recall,

            "f1":
                model_f1,

            "target":
                target,

            "best_model":
                best_model_name

        }

        joblib.dump(
            model_package,
            "trained_model.pkl"
        )

        # =================================================
        # DASHBOARD
        # =================================================

        return render_template(

            "ml_result.html",

            target=target,

            problem_type=problem_type,

            model_name=best_model_name,

            score=model_accuracy,

            mae=(
                mae_value
                if problem_type ==
                "Regression"
                else None
            ),

            precision=model_precision,

            recall=model_recall,

            f1=model_f1,

            feature_importance=
                feature_importance,

            confusion_matrix=
                confusion_matrix_data,

            confusion_labels=
                confusion_labels,

            model_comparison=
                model_comparison,

            best_model=
                best_model_name
        )

    except ValueError as e:

        return f"""
        <h2>⚠️ Dataset Cannot Be Trained</h2>

        <p>
        AutoStream could not create a valid
        machine learning model from this dataset.
        </p>

        <p>
        Try selecting a different target column
        or uploading another structured dataset.
        </p>

        <p>
        Technical details: {e}
        </p>

        <a href="/">Go Back</a>
        """

    except Exception as e:

        return f"""
        <h2>⚠️ Something Went Wrong</h2>

        <p>
        AutoStream could not train the model.
        </p>

        <p>
        Technical details: {e}
        </p>

        <a href="/">Go Back</a>
        """


# =========================================================
# PREDICTION
# =========================================================

@app.route(
    "/predict",
    methods=["GET", "POST"]
)
def predict():

    global trained_model
    global model_features
    global model_problem_type
    global target_encoder

    global model_accuracy
    global model_precision
    global model_recall
    global model_f1
    global best_model_name

    if trained_model is None:

        try:

            package = joblib.load(
                "trained_model.pkl"
            )

            trained_model = (
                package["model"]
            )

            model_features = (
                package["features"]
            )

            model_problem_type = (
                package["problem_type"]
            )

            target_encoder = (
                package["target_encoder"]
            )

            model_accuracy = (
                package.get("accuracy")
            )

            model_precision = (
                package.get("precision")
            )

            model_recall = (
                package.get("recall")
            )

            model_f1 = (
                package.get("f1")
            )

            best_model_name = (
                package.get(
                    "best_model",
                    "Random Forest"
                )
            )

        except:

            return """
            <h2>⚠️ Please Train a Model First</h2>

            <a href="/">Go Back</a>
            """

    if request.method == "GET":

        return render_template(
            "predict.html",
            features=model_features
        )

    try:

        input_data = {}

        for feature in model_features:

            value = request.form.get(
                feature
            )

            if value is None or value == "":

                return f"""
                <h2>⚠️ Missing Value</h2>

                <p>
                Please enter a value for:
                <strong>{feature}</strong>
                </p>

                <a href="/predict">
                Go Back
                </a>
                """

            input_data[feature] = float(
                value
            )

        input_df = pd.DataFrame(
            [input_data]
        )

        prediction = (
            trained_model.predict(
                input_df
            )
        )

        result = prediction[0]

        if (
            model_problem_type ==
            "Classification"
            and target_encoder is not None
        ):

            result = (
                target_encoder
                .inverse_transform(
                    [int(result)]
                )[0]
            )

        result_text = str(result)

        if model_problem_type == "Classification":

            if result_text.upper() == "B":

                prediction_title = (
                    "Benign (B)"
                )

                explanation = (
                    "The model classified "
                    "the entered values as "
                    "Benign."
                )

            elif result_text.upper() == "M":

                prediction_title = (
                    "Malignant (M)"
                )

                explanation = (
                    "The model classified "
                    "the entered values as "
                    "Malignant."
                )

            else:

                prediction_title = (
                    result_text
                )

                explanation = (
                    "The model generated "
                    "this classification."
                )

        else:

            prediction_title = str(
                round(
                    float(result),
                    2
                )
            )

            explanation = (
                "The model generated "
                "this regression prediction."
            )

        model_display = best_model_name

        precision_html = ""

        if model_precision is not None:

            precision_html = f"""

            <div class="metric">

                <div>Precision</div>

                <strong>
                    {model_precision}%
                </strong>

            </div>


            <div class="metric">

                <div>Recall</div>

                <strong>
                    {model_recall}%
                </strong>

            </div>


            <div class="metric">

                <div>F1-Score</div>

                <strong>
                    {model_f1}%
                </strong>

            </div>

            """

        return f"""

<!DOCTYPE html>

<html>

<head>

<title>
AutoStream AI - Prediction Result
</title>

<style>

body {{

    margin: 0;

    font-family: Arial, sans-serif;

    background: #f5f7fb;

    color: #1f2937;

}}

header {{

    background: #111827;

    color: white;

    padding: 20px 50px;

}}

.container {{

    max-width: 850px;

    margin: 60px auto;

    padding: 20px;

}}

.result-box {{

    background: white;

    padding: 45px;

    border-radius: 18px;

    box-shadow:
        0 5px 20px
        rgba(0,0,0,0.08);

    text-align: center;

}}

.icon {{

    font-size: 55px;

}}

.prediction {{

    font-size: 42px;

    font-weight: bold;

    color: #2563eb;

    margin: 25px 0;

}}

.details {{

    display: grid;

    grid-template-columns:
        repeat(2,1fr);

    gap: 15px;

}}

.detail {{

    background: #f8fafc;

    padding: 20px;

    border-radius: 10px;

}}

.metric-grid {{

    display: grid;

    grid-template-columns:
        repeat(3,1fr);

    gap: 15px;

    margin-top: 15px;

}}

.metric {{

    background: #f8fafc;

    padding: 18px;

    border-radius: 10px;

}}

.metric div {{

    color: #64748b;

    font-size: 14px;

    margin-bottom: 7px;

}}

.metric strong {{

    color: #2563eb;

    font-size: 20px;

}}

.explanation {{

    margin-top: 25px;

    padding: 18px;

    background: #eff6ff;

    border-radius: 10px;

    line-height: 1.6;

}}

.buttons {{

    margin-top: 30px;

    display: flex;

    gap: 15px;

    justify-content: center;

    flex-wrap: wrap;

}}

.button {{

    padding: 13px 22px;

    background: #2563eb;

    color: white;

    text-decoration: none;

    border-radius: 8px;

    font-weight: bold;

}}

.secondary {{

    background: #64748b;

}}

@media(max-width:650px) {{

    .details,
    .metric-grid {{

        grid-template-columns: 1fr;

    }}

}}

</style>

</head>


<body>


<header>

<h1>
AutoStream AI
</h1>

</header>


<div class="container">


<div class="result-box">


<div class="icon">

🎯

</div>


<h2>

Prediction Result

</h2>


<div class="prediction">

{prediction_title}

</div>


<div class="details">


<div class="detail">

<div>
Best Model
</div>

<strong>

{model_display}

</strong>

</div>


<div class="detail">

<div>
Accuracy
</div>

<strong>

{model_accuracy}%

</strong>

</div>


</div>


<div class="metric-grid">

{precision_html}

</div>


<div class="explanation">

💡 {explanation}

</div>


<div class="buttons">


<a
href="/predict"
class="button">

🔄 Make Another Prediction

</a>


<a
href="/"
class="button secondary">

📂 Analyze Another File

</a>


</div>


</div>


</div>


</body>

</html>

"""

    except Exception as e:

        return f"""
        <h2>⚠️ Prediction Error</h2>

        <p>{e}</p>

        <a href="/predict">
        Go Back
        </a>
        """


# =========================================================
# RUN APP
# =========================================================

if __name__ == "__main__":

    app.run(debug=True)