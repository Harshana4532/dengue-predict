import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Bidirectional, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Streamlit UI
st.title("Dengue Case Prediction Using LSTM")
uploaded_file = st.file_uploader("Upload your dataset (.csv)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Columns:", df.columns)  # Display columns to check

    # Ensure the correct column names
    df.columns = ["Year", "Month", "Cases", "RainFl", "RainDy", "Temp", "Rhumid"]

    # Convert Year and Month into datetime index
    df["Date"] = pd.to_datetime(df[["Year", "Month"]].assign(Day=1))
    df.set_index("Date", inplace=True)
    df.drop(columns=["Year", "Month"], inplace=True)
    df.fillna(df.median(), inplace=True)

    # Feature Engineering
    lags = 3
    for lag in range(1, lags + 1):
        df[f"RainFl_lag_{lag}"] = df["RainFl"].shift(lag)
        df[f"RainDy_lag_{lag}"] = df["RainDy"].shift(lag)
        df[f"Temp_lag_{lag}"] = df["Temp"].shift(lag)
        df[f"Rhumid_lag_{lag}"] = df["Rhumid"].shift(lag)
    df.dropna(inplace=True)

    # Scale data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    # Prepare sequences
    X = scaled_data[:, 1:]
    y = scaled_data[:, 0]

    def create_sequences(X, y, seq_length=12):
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i+seq_length])
            y_seq.append(y[i+seq_length])
        return np.array(X_seq), np.array(y_seq)

    seq_length = 12
    X_seq, y_seq = create_sequences(X, y, seq_length)

    # Train-test split
    train_size = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:train_size], X_seq[train_size:]
    y_train, y_test = y_seq[:train_size], y_seq[train_size:]

    # Build LSTM model
    model = Sequential([
        Input(shape=(seq_length, X_train.shape[2])),
        LayerNormalization(),
        Bidirectional(LSTM(128, activation="relu", return_sequences=True, kernel_regularizer=l2(0.005))),
        Dropout(0.3),
        Bidirectional(LSTM(64, activation="relu", return_sequences=True, kernel_regularizer=l2(0.005))),
        Dropout(0.3),
        Bidirectional(LSTM(32, activation="relu", kernel_regularizer=l2(0.005))),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

    # Train the model
    early_stopping = EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=400,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )

    # Plot all the plots

    # Training vs Validation Loss
    st.subheader("Training vs Validation Loss")
    fig, ax = plt.subplots()
    ax.plot(history.history["loss"], label="Training Loss")
    ax.plot(history.history["val_loss"], label="Validation Loss")
    ax.legend()
    st.pyplot(fig)

    # Actual vs Predicted Dengue Cases (Training Set)
    y_train_pred = model.predict(X_train)
    y_train_pred_actual = scaler.inverse_transform(np.concatenate([y_train_pred, np.zeros((len(y_train_pred), X.shape[1]))], axis=1))[:, 0]
    y_train_actual = scaler.inverse_transform(np.concatenate([y_train.reshape(-1, 1), np.zeros((len(y_train), X.shape[1]))], axis=1))[:, 0]

    st.subheader("Actual vs Predicted Dengue Cases (Training Set)")
    fig, ax = plt.subplots()
    ax.plot(y_train_actual, label="Actual Dengue Cases (Training)", color="blue")
    ax.plot(y_train_pred_actual, label="Predicted Dengue Cases (Training)", color="red", linestyle="--")
    ax.legend()
    st.pyplot(fig)

    # Actual vs Predicted Dengue Cases (Testing Set)
    y_test_pred = model.predict(X_test)
    y_test_pred_actual = scaler.inverse_transform(np.concatenate([y_test_pred, np.zeros((len(y_test_pred), X.shape[1]))], axis=1))[:, 0]
    y_test_actual = scaler.inverse_transform(np.concatenate([y_test.reshape(-1, 1), np.zeros((len(y_test), X.shape[1]))], axis=1))[:, 0]

    st.subheader("Actual vs Predicted Dengue Cases (Testing Set)")
    fig, ax = plt.subplots()
    ax.plot(y_test_actual, label="Actual Dengue Cases (Testing)", color="blue")
    ax.plot(y_test_pred_actual, label="Predicted Dengue Cases (Testing)", color="red", linestyle="--")
    ax.legend()
    st.pyplot(fig)

    # Forecast for the next 6 months
    forecast_months = 6
    forecast_input = X_seq[-1, :, :]  # Take the last sequence from the training set
    forecast = []
    
    # Generate 6-month forecast
    for _ in range(forecast_months):
        forecast_pred = model.predict(forecast_input.reshape(1, seq_length, X_train.shape[2]))
        forecast.append(forecast_pred[0][0])
        
        # Shift the window for the next prediction
        forecast_input = np.roll(forecast_input, -1, axis=0)
        forecast_input[-1, 0] = forecast_pred[0][0]  # Add the forecasted value

    # Rescale the forecast back to the original scale
    forecast_rescaled = scaler.inverse_transform(np.concatenate([np.array(forecast).reshape(-1, 1), np.zeros((len(forecast), X.shape[1]))], axis=1))[:, 0]

    # Plot the forecast
    st.subheader("6-Month Forecast for Dengue Cases")
    future_dates = pd.date_range(df.index[-1], periods=forecast_months + 1, freq="M")[1:]
    fig, ax = plt.subplots()
    ax.plot(future_dates, forecast_rescaled, label="Forecasted Dengue Cases", color="green", linestyle="--")
    ax.legend()
    st.pyplot(fig)

    # Provide download link
    results_df = pd.DataFrame({"Actual Cases": y_test_actual, "Predicted Cases": y_test_pred_actual})
    csv = results_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")
