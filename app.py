import streamlit as st
import pandas as pd
from graphs import Graphs
from pre_processing import Preprocessing

st.header("Data Preprocessing")
file = st.file_uploader("Enter your CSV or Excel dataset", type=["csv", "xlsx"])

if file is None:
    st.warning("The file is not uploaded")
    st.stop()

# --- Load dataset once ---
try:
    if file.name.endswith(".csv"):
        dataset = pd.read_csv(file)
    else:
        dataset = pd.read_excel(file)
except Exception as e:
    st.error(f"Error reading file: {e}")
    st.stop()

st.success("File uploaded successfully.")
st.write("### Dataset Preview:")
st.dataframe(dataset.head(5))

# create or restore instances in session_state
if "prep" not in st.session_state:
    st.session_state.prep = Preprocessing(dataset.copy())
else:
    # if user uploaded new file, replace instance
    if not st.session_state.prep.dataset.equals(dataset):
        st.session_state.prep = Preprocessing(dataset.copy())

prep = st.session_state.prep
plot = Graphs(dataset)

# ---- Basic tasks (unchanged) ----
task = st.sidebar.selectbox("Basic tasks.", (
    "Please select the option",
    "See your dataset shape",
    "Duplicate rows check",
    "DataType of each column",
    "Unique Value counts of selected column",
    "Checking missing Values and it's percentage",
    "Description of your Dataset"
))

if task == "See your dataset shape":
    st.write("Rows of dataset", dataset.shape[0], "| Columns:", dataset.shape[1])

elif task == "Duplicate rows check":
    dup_count = dataset.duplicated().sum()
    if dup_count > 0:
        st.warning(f"Dataset contains {dup_count} duplicate rows.")
    else:
        st.success("No duplicate rows found.")

elif task == "DataType of each column":
    num_list, cata_list = [], []
    for i in dataset.columns:
        col_dtype = dataset[i].dtype
        if col_dtype in ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]:
            num_list.append(i)
        elif col_dtype in ["object", "category", "bool"]:
            cata_list.append(i)
        else:
            st.write(i, "Unknown dtype:", col_dtype)
    st.write("Numeric Columns:", num_list)
    st.write("Categorical Columns:", cata_list)

elif task == "Unique Value counts of selected column":
    col = st.text_input("Enter the name of the Column", key="unique_col")
    if col:
        if col in dataset.columns:
            st.text(dataset[col].value_counts().to_string())
            st.write(f"There are {dataset[col].nunique()} unique values in {col}")
        else:
            st.error(f"The column '{col}' does not exist.")
    else:
        st.warning("Please enter the column name before proceeding")

elif task == "Checking missing Values and it's percentage":
    missing = dataset.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        for col, cnt in missing.items():
            pct = (cnt / len(dataset)) * 100
            st.warning(f"{col}: {cnt} missing ({pct:.2f}%)")
    else:
        st.success("No missing values found.")

elif task == "Description of your Dataset":
    st.write(dataset.describe(include='all'))

# ---- Graphs and Plots (unchanged usage) ----
graph = st.sidebar.selectbox("Graphs and Plots", (
    "Please select the option", "Histogram", "Box Plot", "Bar Chart", "Scatter Plot", "Heatmap", "Pair Plot"
))

if graph == "Histogram":
    temp_task = st.radio("Histogram", ["Wanna see all numeric features graph", "Particular Column graph"], key="hist_radio")
    plot.Histogram(dataset, temp_task)
elif graph == "Box Plot":
    temp_task = st.radio("Box Plot", ["Wanna see all numeric features graph", "Particular Column graph"], key="box_radio")
    plot.BoxPlot(dataset, temp_task)
elif graph == "Bar Chart":
    plot.BarPlot(dataset)
elif graph == "Scatter Plot":
    plot.ScatterPlot(dataset)
elif graph == "Heatmap":
    plot.HeatMap(dataset)
elif graph == "Pair Plot":
    plot.PairPlot(dataset=dataset)

# ---- Preprocessing controls ----
st.sidebar.markdown("---")
st.sidebar.write("### Preprocessing")

# --- TRAIN-TEST SPLIT: input outside button ---
target = st.sidebar.text_input("Enter the target feature (column name):", key="target_name")
if st.sidebar.button("Run Train Test Split"):
    if not target:
        st.sidebar.error("Enter target name first.")
    elif target not in dataset.columns:
        st.sidebar.error("Target not found in dataset.")
    else:
        # call method that accepts target argument
        try:
            prep.TrainTestSplit(target)   # ensure your class method signature is TrainTestSplit(self, target_name)
            st.success("Train/Test split done.")
            st.write("X_train shape:", prep.X_train.shape if prep.X_train is not None else None)
        except Exception as e:
            st.error(f"TrainTestSplit error: {e}")

# --- Missing values (button) ---
if st.sidebar.button("Handle Missing Values"):
    try:
        prep.HandlingMissingValues()
    except Exception as e:
        st.error(f"Missing values handler error: {e}")

# --- Remove duplicates ---
if st.sidebar.button("Remove Duplicates"):
    try:
        prep.RemovingDuplicates()
    except Exception as e:
        st.error(f"Remove duplicates error: {e}")

# --- Encoding ---
enc_type = st.sidebar.selectbox("Encoding method", ("Label Encoding", "One-Hot Encoding"))
if st.sidebar.button("Run Encoding"):
    if prep.X_train is None:
        st.sidebar.error("Run Train-Test Split first.")
    else:
        try:
            prep.Encoding(encoding_type=enc_type)  # ensure method signature supports arg or default mapping
            st.success("Encoding done.")
        except Exception as e:
            st.error(f"Encoding error: {e}")

# --- Scaling ---
scale_type = st.sidebar.selectbox("Scaling method", ("StandardScaler", "MinMaxScaler"))
if st.sidebar.button("Run Scaling"):
    if prep.X_train is None:
        st.sidebar.error("Run Train-Test Split first.")
    else:
        try:
            prep.FeatureScaling(scaling_type=scale_type)
            st.success("Scaling done.")
        except Exception as e:
            st.error(f"Scaling error: {e}")

# --- Outlier handling ---
if st.sidebar.button("Run Outlier Handling"):
    if prep.X_train is None:
        st.sidebar.error("Run Train-Test Split first.")
    else:
        try:
            prep.OutlierHandling()
            st.success("Outlier handling done.")
        except Exception as e:
            st.error(f"Outlier handling error: {e}")
