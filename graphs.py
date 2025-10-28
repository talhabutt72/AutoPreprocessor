import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt


class Graphs:
    
    def __init__(self, dataset) -> None:
        self.dataset = dataset
    
    def Histogram(self, dataset, temp_task):
        
        st.subheader("Histogram is used for observing the distribuation on the values.")
        
        if temp_task == "Wanna see all numeric features graph":
        
            for col in dataset.select_dtypes(include=['int64', 'float64']).columns:
                fig, ax = plt.subplots(figsize=(15, 15))
                sns.histplot(dataset[col], kde=True)
                plt.title(f"Distribution of {col}")
                st.pyplot(fig)
        else:
            
            feature_name = st.text_input("Please enter the column name: ")
            
            if feature_name:
                if feature_name in dataset.columns:
                    if dataset[feature_name].dtype in ["int8", "int64", "float64"]:
                        fig, ax = plt.subplots(figsize=(15, 15))
                        sns.histplot(dataset[feature_name], kde=True, ax=ax)
                        ax.set_title(f"Distribution of {feature_name}")
                        st.pyplot(fig)
                    else:
                        st.warning("Please enter a numeric column.")
                else:
                    st.error("Column name not found in dataset.")
            else:
                st.info("Waiting for column name input...")
    
    def BoxPlot(self, dataset, temp_task):
        
        st.subheader("BoxPlot is used for Outlier detection.")
        
        if temp_task == "Wanna see all numeric features graph":
        
            for col in dataset.select_dtypes(include=['int64', 'float64']).columns:
                fig, ax = plt.subplots()
                sns.boxplot(dataset[col])
                plt.title(f"Boxplot of {col}")
                st.pyplot(fig)
        else:
            
            feature_name = st.text_input("Please enter the column name: ")
            
            if feature_name:
                if feature_name in dataset.columns:
                    if dataset[feature_name].dtype in ["int8", "int64", "float64"]:
                        fig, ax = plt.subplots(figsize=(15, 15))
                        sns.boxplot(dataset[feature_name])
                        ax.set_title(f"Boxplot of {feature_name}")
                        st.pyplot(fig)
                    else:
                        st.warning("Please enter a numeric column.")
                else:
                    st.error("Column name not found in dataset.")
            else:
                st.info("Waiting for column name input...")
    
    def BarPlot(self, dataset):
        
        st.subheader("BarPlot is used for Category vs count/value.")
        
        x = st.text_input("Please enter the Catagorical column name: ")
        y = st.text_input("Please enter the Numerical column name: ")

        
        if not x or not y:
            
            st.info("Waiting for column name input...")
            
        else:
            
            if x in dataset.columns and y in dataset.columns:
                    if dataset[x].dtype in ["object", "category"] and  dataset[y].dtype in ["int8", "int64", "float64"]:
                    
                        
                            fig, ax = plt.subplots(figsize=(15, 6))
                            sns.barplot(data = dataset, x = x, y = y )
                            ax.set_title(f"BarPlot of {x} and {y}")
                            st.pyplot(fig)
                    else:
                        st.warning("Please enter catagorical column in and Numeric column name in y for getting your BarPlot.")
            else:
                    st.error("Column name not found in dataset.")
    
    def ScatterPlot(self, dataset):
        
        st.subheader("Scatter Plot is used for relationship btw 2 numeric columns.")
        
        x = st.text_input("Please enter the X (Numeric) column name: ")
        y = st.text_input("Please enter the Y (Numeric) column name: ")

        if not x or not y:
            
            st.info("Waiting for column name input...")
            
        else:
            
            if x in dataset.columns and y in dataset.columns:
                    if dataset[x].dtype.kind in 'if' and dataset[y].dtype.kind in 'if':
                    
                            fig, ax = plt.subplots(figsize=(15, 15))
                            sns.scatterplot(data = dataset, x = x, y = y )
                            ax.set_title(f"Scatter Plot of {x} vs {y}")
                            st.pyplot(fig)
                    else:
                        st.warning("Please enter Numerical column in X and Y for getting your ScatterPlot.")
            else:
                    st.error("Column name not found in dataset.")
    
    def HeatMap(self, dataset):
        
        st.subheader("Heatmap Plot is used for relationship btw numeric features.")
        
        col_dtye = []
        
        for column in dataset.columns:
            
            if dataset[column].dtype in ["object", "category"]:
                col_dtye.append(column)
                
        
        if not col_dtye:
            st.write("You dataset have only numeric features, so here is your heatmap.")
        else:
            st.warning("Your dataset have catagorical featues, but heatmap is work only on numeric features, if you wanna drop that features and see the heatmap??")
            
            temp_ = st.radio("Please Select.", ["Yes", "No"])
            
            if temp_ == "Yes":
                
                for col in col_dtye:
                    dataset = dataset.drop(col, axis = 1)
                
                fig, ax = plt.subplots(figsize=(15, 15))
                sns.heatmap(data=dataset.corr(), annot=True, cmap="coolwarm", fmt=".2f", square=True)
                st.pyplot(fig)
            
            else:
                st.write("You can use other graphs for plotting.")
    
    def PairPlot(self, dataset):
    
        
        st.subheader("Pairplot Plot is used for relationship btw numeric features and target feature.")
        
        target = st.text_input("Please enter the target Variable.")
        
        if not target:
            
            st.info("Waiting for column name input...")
            
        else:
            
            if target in dataset.columns:
                numeric_data = dataset.select_dtypes(include=['number'])
                if target not in numeric_data.columns:
                    numeric_data[target] = dataset[target]
                
                sns.pairplot(data=numeric_data, hue=target)
                st.pyplot(plt.gcf())
                
            else:
                    st.error("Column name not found in dataset.")
        

                
                
        
        
            
        
        

