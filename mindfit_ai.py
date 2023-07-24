import pandas
import seaborn
import matplotlib.pyplot as plot
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np


class mindfit:
    """Machine learning Model for predicting DALY with Mental Health Insights"""

    def __init__(self) -> None:
        """Load, preprocess the datasets, inform after completion"""

        self.get_dataset()
        self.preprocess_dataset()
        print("✔ Ready to build.")

    def get_dataset(self) -> None:
        """Reads the dataset from the datasets/ directory"""

        # Load datasets separately
        dataset_1 = pandas.read_csv(
            "datasets/mental-and-substance-use-as-share-of-disease.csv"
        )
        dataset_2 = pandas.read_csv(
            "datasets/prevalence-by-mental-and-substance-use-disorder.csv"
        )

        # Merge the features of two datasets
        self.dataset = dataset_2.merge(right=dataset_1, how="inner")
        print("✔ Loaded Datasets.")

    def preprocess_dataset(self, inplace=True) -> pandas.DataFrame | None:
        """Preprocess the data for cleaner processing in future

        Args:
            inplace (bool, optional): If want to make changes in object itself. Defaults to True.

        Returns:
            DataFrame: A copy of dataset which is preprocessed
        """
        temp = self.dataset.set_axis(
            labels=[
                "Country",
                "Code",
                "Year",
                "Schizophrenia disorders",
                "Bipolar disorders",
                "Eating disorders",
                "Anxiety disorders",
                "Drug use disorders",
                "Depressive disorders",
                "Alcohol use disorders",
                "DALY",
            ],
            axis=1,
        )

        # Drop the rows containing null values
        temp.dropna(inplace=True)

        # Drop Country Codes
        temp.drop(columns=["Code"], inplace=True)
        self.countries = list(temp["Country"].drop_duplicates())
        self.country_encoder = LabelEncoder()
        self.country_encoder.fit(y=temp["Country"])
        temp["Country"] = self.country_encoder.transform(temp["Country"])

        # Return the new dataset (if not inplace)
        if inplace:
            self.dataset = temp
            print("✔ Data Preprocessed.")
        else:
            return temp

    def build(self) -> None:
        """Builds the model"""

        # Split columns as features (X) and label {y}
        X = self.dataset.drop(columns=["DALY"])
        y = self.dataset["DALY"]

        # Split the rows as train and test records
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Build the model
        self.model = RandomForestRegressor()
        self.model.fit(X_train, y_train)

        # Prompt the success
        print("✔ Model Build Successfully.")

        # Evaluate the model based on mae, mse, r-mse and r2score metrics
        y_predict = self.model.predict(X_test)
        mae = round(mean_absolute_error(y_test, y_predict), 2)
        mse = round(mean_squared_error(y_test, y_predict), 2)
        rmse = round(np.sqrt(mse), 2)
        r2s = round(r2_score(y_test, y_predict), 2)

        # Print the evaluation of the built model
        print(
            f"""🡲 Model Evaluation (Metrics)
        1. Mean Absolute Error: {mae}
        2. Mean Squared Error: {mse}
        3. Root Mean Squared Error: {rmse}
        4. R2 Score: {r2s} {'✅ Success!' if r2s >= 0.9 else '⚠️ Unexpected!'}"""
        )

    def predict(
        self: str,
        country: int,
        year: int,
        schizophrenia: float,
        bipolar: float,
        eating: float,
        anxiety: float,
        drug_use: float,
        depressive: float,
        alcohol_use: float,
    ) -> None:
        """Prints prediction of DALY presentation based on given parameters"""

        # Get the country
        if country not in self.countries:
            print(f"⚠️ Country '{country}' is not known to us.")
            return
        country = self.country_encoder.transform(["Afghanistan"])[0]

        # Make prediction
        sample = pandas.DataFrame(
            columns=[
                "Country",
                "Year",
                "Schizophrenia disorders",
                "Bipolar disorders",
                "Eating disorders",
                "Anxiety disorders",
                "Drug use disorders",
                "Depressive disorders",
                "Alcohol use disorders",
            ],
            data=[
                [
                    country,
                    year,
                    schizophrenia,
                    bipolar,
                    eating,
                    anxiety,
                    drug_use,
                    depressive,
                    alcohol_use,
                ]
            ],
        )

        # Print Result
        daly = round(self.model.predict(sample)[0], 3)
        print(f"🡲 DALYs (Disability-Adjusted Life Years): {daly}%.")
        print("  on overall population.")

    def plot_relation(self, condition: str) -> None:
        """Based on given condition/feature, shows how it effects DALY percentage

        Args:
            condition (str): any feature value "Schizophrenia disorders",
            "Bipolar disorders", "Eating disorders", "Anxiety disorders",
            "Drug use disorders", "Depressive disorders", "Alcohol use disorders"
        """
        plot.figure(figsize=(16, 9))
        seaborn.jointplot(
            data=self.dataset,
            x=condition,
            y="DALY",
            kind="hex",
            marginal_ticks=True,
            marginal_kws={"bins": 30, "fill": False},
        )

        # Save the graph in current directory
        plot.savefig(f"{condition} impact on DALY.png")


if __name__ == "__main__":
    # Load the model
    ai = mindfit()

    # Build the model
    ai.build()

    # Test with random sample
    ai.predict(
        "Afghanistan",
        1990,
        0.22320578,
        0.70302314,
        0.12770003,
        4.713314,
        0.45,
        4.996118,
        0.44,
    )
    
    # Plot visualization for random feature
    ai.plot_relation("Alcohol use disorders")
