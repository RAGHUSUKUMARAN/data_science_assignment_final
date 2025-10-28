# ---- Scaling + Prediction block ----
predict_btn = st.button("Predict Survival")

# numeric columns you used at train time (very likely these)
num_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

if predict_btn:
    try:
        # 1) Ensure numeric columns exist
        for c in num_cols:
            if c not in input_df.columns:
                raise KeyError(f"Missing numeric column in input: {c}")

        # 2) Try a straightforward scaling first
        try:
            input_df[num_cols] = scaler.transform(input_df[num_cols])
        except ValueError as ve:
            # If scaler complains about feature names/order, attempt to auto-align
            st.warning("Scaler rejected input shape/order — attempting to auto-align features...")
            scaler_cols = getattr(scaler, "feature_names_in_", None)
            if scaler_cols is None:
                # fallback: try to use the expected numeric columns list
                scaler_cols = num_cols
            scaler_cols = list(scaler_cols)

            # Add any missing columns (with zeros) so shapes match
            for c in scaler_cols:
                if c not in input_df.columns:
                    input_df[c] = 0

            # Reorder and transform
            input_df[scaler_cols] = scaler.transform(input_df[scaler_cols])

            # If scaler had extra columns not in our model input, keep them but we'll later select model features
            st.info(f"Used scaler columns: {scaler_cols}")

        # 3) Prepare the final feature set in the exact order model expects (if available)
        model_cols = getattr(model, "feature_names_in_", None)
        if model_cols is not None:
            model_cols = list(model_cols)
            # ensure all model_cols exist in input_df, add missing with 0
            for c in model_cols:
                if c not in input_df.columns:
                    input_df[c] = 0
            input_for_model = input_df[model_cols]
        else:
            # fallback — assume our input_df columns are in the correct order
            input_for_model = input_df.copy()

        # 4) Predict
        pred = model.predict(input_for_model)[0]
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_for_model)[0][1]  # prob of survival (class 1)
        elif hasattr(model, "decision_function"):
            # not probability but at least a confidence-like score
            score = model.decision_function(input_for_model)[0]
            proba = None
        else:
            proba = None

        # 5) Display nicely
        label = "Survived" if pred == 1 else "Did not survive"
        if proba is not None:
            st.success(f"Prediction: **{label}** — survival probability: **{proba:.2%}**")
        else:
            st.success(f"Prediction: **{label}**")
        st.write("### Input used for prediction (after scaling / alignment)")
        st.dataframe(input_for_model)

    except KeyError as ke:
        st.error(f"Input error: {ke}")
    except Exception as e:
        # show debugging info for the common ValueError about feature names (helps fix the mismatch)
        st.error("An error occurred during scaling/prediction.")
        st.exception(e)
        # helpful debug info:
        scaler_names = getattr(scaler, "feature_names_in_", None)
        model_names = getattr(model, "feature_names_in_", None)
        st.write("**Debug info:**")
        st.write("- Input columns:", list(input_df.columns))
        st.write("- Scaler.feature_names_in_:", list(scaler_names) if scaler_names is not None else "N/A")
        st.write("- Model.feature_names_in_:", list(model_names) if model_names is not None else "N/A")
