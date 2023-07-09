import pandas
import seaborn
import matplotlib.pyplot as plot
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np




class mindfit:
    def __init__(self) -> None:
        self.get_dataset()
        self.preprocess_dataset()
        print("✔ Ready to build.")
        print("  Hint: use build() method.")
        

    def get_dataset(self):
        dataset_1 = pandas.read_csv(
            "datasets/mental-and-substance-use-as-share-of-disease.csv")
        dataset_2 = pandas.read_csv(
            "datasets/prevalence-by-mental-and-substance-use-disorder.csv")
        self.dataset = dataset_2.merge(right=dataset_1, how="inner")
        print("✔ Loaded Datasets.")

    def preprocess_dataset(self, inplace=True):
        temp = self.dataset.set_axis(
            labels=["Country", "Code", "Year", "Schizophrenia disorders",
                    "Bipolar disorders", "Eating disorders",
                    "Anxiety disorders", "Drug use disorders",
                    "Depressive disorders", "Alcohol use disorders",
                    "DALY"],
            axis=1)

        # Drop the rows containing null values
        temp.dropna(inplace=True)

        # Drop Country Codes
        temp.drop(columns=["Code"], inplace=True)
        self.countries = list(temp["Country"].drop_duplicates())

        self.country_encoder = LabelEncoder()
        self.country_encoder.fit(y=temp["Country"])
        temp["Country"] = self.country_encoder.transform(temp["Country"])
        

        if inplace:
            self.dataset = temp
            print("✔ Data Preprocessed.")
        else:
            return temp
        
    def build(self):
        X = self.dataset.drop(columns=["DALY"])
        y = self.dataset["DALY"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self.model = RandomForestRegressor()
        self.model.fit(X_train, y_train)
        print("✔ Model Build Successfully")
        y_predict = self.model.predict(X_test)
        mae = round(mean_absolute_error(y_test, y_predict), 2)
        mse = round(mean_squared_error(y_test, y_predict), 2)
        rmse = round(np.sqrt(mse), 2)
        r2s = round(r2_score(y_test, y_predict), 2)
        print(f"""🡲 Model Evaluation (Metrics)
        1. Mean Absolute Error: {mae}
        2. Mean Squared Error: {mse}
        3. Root Mean Squared Error: {rmse}
        4. R2 Score: {r2s} {'✅ Success!' if r2s >= 0.9 else '⚠️ Unexpected!'}""")
        
    def predict(self, country, year, schizophrenia, bipolar, eating,
                    anxiety, drug_use,
                    depressive, alcohol_use):
        if country not in self.countries:
            print(f"⚠️ Country '{country}' is not known to us.")
            return
        
        country = self.country_encoder.transform(["Afghanistan"])[0]

        sample = pandas.DataFrame(columns=["Country", "Year", "Schizophrenia disorders",
            "Bipolar disorders", "Eating disorders",
            "Anxiety disorders", "Drug use disorders",
            "Depressive disorders", "Alcohol use disorders"], data=[[country, year, schizophrenia, bipolar, eating,
            anxiety, drug_use,
            depressive, alcohol_use]])
        daly = round(self.model.predict(sample)[0], 3)
        print(f"🡲 DALYs (Disability-Adjusted Life Years): {daly}%.")
        print("  on overall population.")
        
    def plot_relation(self, condition):
        plot.figure(figsize=(16, 9))
        seaborn.jointplot(data=self.dataset, 
                        x=condition, y="DALY", 
                        kind="hex", 
                        marginal_ticks=True, marginal_kws={"bins":30, "fill":False})
        plot.savefig(f"{condition} impact on DALY.png")

        
if __name__ == "__main__":
    ai = mindfit()
    ai.build()
    ai.predict("Afghanistan", 1990,0.22320578,0.70302314,0.12770003,4.713314,0.45,4.996118,0.44)
    ai.plot_relation("Alcohol use disorders")