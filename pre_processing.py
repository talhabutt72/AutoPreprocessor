import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

class Preprocessing:
    def __init__(self, dataset):
        self.dataset = dataset
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.encoders = {}
        self.scaler = None

    # ------------------- Missing Values -------------------
    def HandlingMissingValues(self):
        dataset = self.dataset.copy()

        if dataset.isnull().sum().sum() == 0:
            st.success("No missing values found in the dataset.")
            return dataset

        st.warning(" Missing values detected in the dataset.")
        st.write(dataset.isnull().sum())

        strategy = st.radio("Select fill strategy for numeric columns:", ["Mean", "Median"])

        num_cols = dataset.select_dtypes(include=['int64', 'float64']).columns
        for col in num_cols:
            if dataset[col].isnull().sum() > 0:
                if strategy == "Mean":
                    dataset[col].fillna(dataset[col].mean(), inplace=True)
                else:
                    dataset[col].fillna(dataset[col].median(), inplace=True)

        cat_cols = dataset.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            if dataset[col].isnull().sum() > 0:
                dataset[col].fillna(dataset[col].mode()[0], inplace=True)

        st.success("Missing values handled successfully.")
        self.dataset = dataset
        return dataset

    # ------------------- Duplicate Removal -------------------
    def RemovingDuplicates(self):
        dataset = self.dataset.copy()
        dup_count = dataset.duplicated().sum()
        if dup_count > 0:
            st.warning(f" Dataset contains {dup_count} duplicate rows.")
            dataset.drop_duplicates(inplace=True)
            st.success(" Duplicate rows removed.")
        else:
            st.success(" No duplicate rows found.")
        self.dataset = dataset
        return dataset

    # ------------------- Train-Test Split -------------------
    def TrainTestSplit(self, target_name):
        dataset = self.dataset.copy()

        if target_name not in dataset.columns:
            st.error("❌ Target column not found in dataset.")
            return None

        X = dataset.drop(target_name, axis=1)
        y = dataset[target_name]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        st.success(f" Train-test split completed. Train shape: {self.X_train.shape}, Test shape: {self.X_test.shape}")
        return self.X_train, self.X_test, self.y_train, self.y_test

    # ------------------- Encoding -------------------
    def Encoding(self, encoding_type="Label Encoding"):
        if self.X_train is None or self.X_test is None:
            st.error("❌ Run Train-Test Split before encoding.")
            return None

        cat_cols = self.X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        if len(cat_cols) == 0:
            st.info("ℹ️ No categorical columns found.")
            return self.X_train, self.X_test

        st.write("Categorical Columns:", cat_cols)

        if encoding_type == "Label Encoding":
            for col in cat_cols:
                le = LabelEncoder()
                self.X_train[col] = le.fit_transform(self.X_train[col].astype(str))
                # handle unseen categories
                self.X_test[col] = self.X_test[col].astype(str).map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
                self.encoders[col] = le
            st.success("✅ Label Encoding applied.")

        else:
            n_train = len(self.X_train)
            combined = pd.concat([self.X_train, self.X_test])
            combined = pd.get_dummies(combined, columns=cat_cols, drop_first=True)
            self.X_train = combined.iloc[:n_train, :].copy()
            self.X_test = combined.iloc[n_train:, :].copy()
            st.success("✅ One-Hot Encoding applied.")

        return self.X_train, self.X_test

    # ------------------- Feature Scaling -------------------
    def FeatureScaling(self, scaling_type="StandardScaler"):
        if self.X_train is None or self.X_test is None:
            st.error("❌ Run Train-Test Split before scaling.")
            return None

        num_cols = self.X_train.select_dtypes(include=['int64', 'float64']).columns
        if len(num_cols) == 0:
            st.info("No numeric columns found for scaling.")
            return self.X_train, self.X_test

        self.scaler = StandardScaler() if scaling_type == "StandardScaler" else MinMaxScaler()

        self.X_train[num_cols] = self.scaler.fit_transform(self.X_train[num_cols])
        self.X_test[num_cols] = self.scaler.transform(self.X_test[num_cols])
        
        st.success(f"{scaling_type} applied successfully.")
        return self.X_train, self.X_test

    # ------------------- Outlier Handling -------------------
    def OutlierHandling(self):
        if self.X_train is None:
            st.error("❌ Run Train-Test Split before outlier handling.")
            return None

        num_cols = self.X_train.select_dtypes(include=['int64', 'float64']).columns

        for col in num_cols:
            Q1 = self.X_train[col].quantile(0.25)
            Q3 = self.X_train[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            self.X_train[col] = self.X_train[col].clip(lower, upper)
            self.X_test[col] = self.X_test[col].clip(lower, upper)

        st.success("Outlier handling applied using IQR clipping.")
        return self.X_train, self.X_test
