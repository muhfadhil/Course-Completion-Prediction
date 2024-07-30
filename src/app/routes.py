from flask import request, render_template, jsonify
from . import app
from .models import load_model


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")
    elif request.method == "POST":
        CourseCategory = request.form["course-category"]
        TimeSpentOnCourse = int(request.form["time-spent"])
        NumberOfVideosWatched = int(request.form["videos-watched"])
        NumberOfQuizzesTaken = int(request.form["quizzes-taken"])
        QuizScores = int(request.form["quiz-scores"])
        CompletionRate = int(request.form["completion-rate"])
        DeviceType = int(request.form["device-type"])
        Model = request.form["model-choice"]

        course_category_val = ["Arts", "Business", "Health", "Programming", "Science"]
        course_category = [1 if i == CourseCategory else 0 for i in course_category_val]

        data_input = [
            TimeSpentOnCourse,
            NumberOfVideosWatched,
            NumberOfQuizzesTaken,
            QuizScores,
            CompletionRate,
            DeviceType,
            *course_category,
        ]

        model = load_model(Model)
        prediction_map = {0: "Not Complete", 1: "Complete"}
        prediction = model.predict([data_input])[0]
        prediction = prediction_map[prediction]

        model_map = {
            "lr": "Logistic Regression",
            "dt": "Decision Tree",
            "rf": "Random Forest",
            "xgb": "XGBoost Classifier",
        }

        device_map = {0: "Dekstop", 1: "Mobile"}

        return render_template(
            "index.html",
            model=model_map[Model],
            prediction=prediction,
            course_category=CourseCategory,
            time_spent=TimeSpentOnCourse,
            videos_watched=NumberOfVideosWatched,
            quizzes_taken=NumberOfQuizzesTaken,
            quiz_scores=QuizScores,
            completion_rate=CompletionRate,
            device_type=device_map[DeviceType],
        )
