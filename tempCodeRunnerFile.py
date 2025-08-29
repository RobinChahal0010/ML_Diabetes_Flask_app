@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        file = request.files['file']
        if not file:
            return redirect(url_for('home'))

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        df = pd.read_csv(filepath)
        X = df[FEATURE_NAMES]
        df["Prediction"] = model.predict(X)
        df["Prediction"] = df["Prediction"].map({0: "Not Diabetic", 1: "Diabetic"})

        output_path = os.path.join(app.config['UPLOAD_FOLDER'], "batch_predictions.csv")
        df.to_csv(output_path, index=False)

        return send_file(output_path, as_attachment=True)
    except Exception as e:
        return f"Error in batch prediction: {e}"
