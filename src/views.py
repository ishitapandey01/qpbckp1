# Import packages
import os
from datetime import datetime

import flask
import numpy as np
import pandas as pd
from flask import render_template, request, session
from werkzeug.utils import secure_filename

from src import app
from src.objective import ObjectiveTest
from src.subjective import SubjectiveTest
from src.utils import backup, relative_ranking

# Placeholders
global_answers = list()


@app.route('/')
@app.route('/home')
def home():
    ''' Renders the home page '''
    directory = os.path.join(str(os.getcwd()), "database")
    session["database_path"] = os.path.join(directory, "userlog.csv")
    if "userlog.csv" not in os.listdir(directory):
        df = pd.DataFrame(columns=["DATE", "USERNAME", "SUBJECT", "SUBJECT_ID", "TEST_TYPE", "TEST_ID", "SCORE", "RESULT"])
        df.to_csv(session["database_path"], index=False)
    session["date"] = datetime.now()
    return render_template(
        "index.html",
        date=session["date"].day,
        month=session["date"].month,
        year=session["date"].year
    )


@app.route("/form", methods=['GET', 'POST'])
def form():
    ''' Prompt user to start the test '''
    session["username"] = request.form["username"] if request.form["username"] else "Username"
    return render_template("form.html", username=session["username"])


@app.route("/generate_test", methods=["GET", "POST"])
def generate_test():
    session["subject_id"] = request.form["subject_id"]

    # Subject selection
    subject_map = {
        "0": ("SOFTWARE ENGINEERING", "software-testing.txt"),
        "1": ("DBMS", "dbms.txt"),
        "2": ("Machine Learning", "ml.txt")
    }

    if session["subject_id"] in subject_map:
        session["subject_name"], file_name = subject_map[session["subject_id"]]
        session["filepath"] = os.path.join(os.getcwd(), "corpus", file_name)
    elif session["subject_id"] == "99":
        file = request.files["file"]
        session["filepath"] = secure_filename(file.filename)
        file.save(session["filepath"])
        session["subject_name"] = "CUSTOM"
    else:
        return "Invalid subject ID", 400

    session["test_id"] = request.form["test_id"]

    if session["test_id"] == "0":
        # Objective test
        objective_generator = ObjectiveTest(session["filepath"])
        question_list, answer_list = objective_generator.generate_test()
        global_answers.extend(answer_list)

        return render_template(
            "objective_test.html",
            username=session["username"],
            testname=session["subject_name"],
            question1=question_list[0],
            question2=question_list[1],
            question3=question_list[2]
        )

    elif session["test_id"] == "1":
        # Subjective test
        subjective_generator = SubjectiveTest(session["filepath"])
        question_list, answer_list = subjective_generator.generate_test(num_questions=2)
        global_answers.extend(answer_list)

        return render_template(
            "subjective_test.html",
            username=session["username"],
            testname=session["subject_name"],
            question1=question_list[0],
            question2=question_list[1]
        )

    return "Invalid test ID", 400


@app.route("/output", methods=["GET", "POST"])
def output():
    default_ans = [str(x).strip().upper() for x in global_answers]
    user_ans = []

    if session["test_id"] == "0":
        # Objective answers
        user_ans = [
            str(request.form.get("answer1", "")).strip().upper(),
            str(request.form.get("answer2", "")).strip().upper(),
            str(request.form.get("answer3", "")).strip().upper()
        ]
    elif session["test_id"] == "1":
        # Subjective answers
        user_ans = [
            str(request.form.get("answer1", "")).strip().upper(),
            str(request.form.get("answer2", "")).strip().upper()
        ]
    else:
        return "Invalid test type", 400

    total_score = 0
    status = None

    if session["test_id"] == "0":
        for i in range(min(len(user_ans), len(default_ans))):
            if user_ans[i] == default_ans[i]:
                total_score += 100
        total_score = round(total_score / 3, 3)
        status = "Pass" if total_score >= 33.33 else "Fail"

    elif session["test_id"] == "1":
        subjective_generator = SubjectiveTest(session["filepath"])
        for i in range(min(len(default_ans), len(user_ans))):
            total_score += subjective_generator.evaluate_subjective_answer(default_ans[i], user_ans[i])
        total_score = round(total_score / 2, 3)
        status = "Pass" if total_score > 50.0 else "Fail"

    session["score"] = total_score
    session["result"] = status

    try:
        backup(session)
    except Exception as e:
        print("Exception raised at `views.output`: ", e)

    max_score, min_score, mean_score = relative_ranking(session)
    global_answers.clear()

    return render_template(
        "output.html",
        show_score=session["score"],
        username=session["username"],
        subjectname=session["subject_name"],
        status=session["result"],
        max_score=max_score,
        min_score=min_score,
        mean_score=mean_score
    )
