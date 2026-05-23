import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Customer Personality Analysis
    """)
    return


@app.cell
def _():
    import math
    import pandas as pd
    import numpy as np
    import kagglehub
    import matplotlib.pyplot as plt
    import seaborn as sns
    from datetime import datetime
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import statsmodels.api as sm

    return (
        datetime,
        kagglehub,
        math,
        pd,
        plt,
        sm,
        sns,
        variance_inflation_factor,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Uploading a dataset
    """)
    return


@app.cell
def _(kagglehub, pd):
    dataset_path = kagglehub.dataset_download("imakash3011/customer-personality-analysis")

    df = pd.read_csv(dataset_path + "/marketing_campaign.csv", delimiter="\t")
    return (df,)


@app.cell
def _(df):
    df.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data Analysis
    """)
    return


@app.cell
def _(df):
    df.info()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## NaNs
    """)
    return


@app.cell
def _(df):
    df.isnull().sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `In several rows, the Income field contains NaN, which may indicate that users have not filled in this field in the information.`
    """)
    return


@app.cell
def _(df):
    df[df.isnull().any(axis=1)]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `The data with NaN is diverse, and you cannot fill the Income field with the sample median. You can't average different users in Income. Let's leave it as it is for now.`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Current date
    """)
    return


@app.cell
def _(df, pd):
    pd.to_datetime(df["Dt_Customer"], format="%d-%m-%Y").max()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `Since the dataset does not specify the date of data collection, I will assume that the current date is 2014-06-29.`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dataset statistics
    """)
    return


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(df):
    df.columns
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Graphs
    """)
    return


@app.cell
def _(df, math, plt, sns):
    def generate_graphs(columns: str, col_numb=3):
        n_cols = len(columns)
        n_col_numb = col_numb
        rows = math.ceil(n_cols / n_col_numb)

        fig, axes = plt.subplots(nrows=2*rows, ncols=n_col_numb, 
                                 figsize=(5*n_col_numb, 5*rows),
                                 gridspec_kw={'hspace': 0.35})


        for i, name_col in enumerate(columns):
            row = i // n_col_numb
            col = i % n_col_numb
            sns.histplot(df[name_col], kde=True, ax=axes[row*2, col])
            sns.boxenplot(data=df[name_col], orient="h", ax=axes[1+row*2, col])

            axes[row*2, col].set_xlabel("")
            axes[row*2, col].set_title(name_col)

            axes[1+row*2, col].set_xlabel("")


        for j in range(n_cols, n_col_numb * rows):
            row = j // n_col_numb
            col = j % n_col_numb
            fig.delaxes(axes[row*2, col])
            fig.delaxes(axes[1+row*2, col])

        plt.show()

    return (generate_graphs,)


@app.cell
def _(plt, sns):
    def generate_pie(values: list, 
                     indexes: list, 
                     explode: list, 
                     pos: int, 
                     loc: str, 
                     bbox: tuple, 
                     title: str):
        plt.subplot(3, 2, pos)

        plt.pie(
            x=values,
            labels=indexes,
            colors=sns.color_palette("hls", len(values)),
            autopct='%.0f%%',
            shadow=True,
            explode=explode
        )
        plt.title(title)
        plt.legend(labels=[f"{name}: {value}" for name, value in zip(indexes, values)], 
                   loc=loc, 
                   bbox_to_anchor=bbox)

    return (generate_pie,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Distribution personal numerical features
    """)
    return


@app.cell
def _(generate_graphs):
    clomns_to_plot_personal = ['Year_Birth', 'Income', 'Recency']
    generate_graphs(clomns_to_plot_personal)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `There are people who are information around 100 years old or more. There is one outlier or abnormally large value in Income.`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Distribution categorial features
    """)
    return


@app.cell
def _(df):
    education = df["Education"].value_counts()
    martial = df["Marital_Status"].value_counts()
    kidhome = df["Kidhome"].value_counts()
    teenhome = df["Teenhome"].value_counts()

    customer_dt_year = df["Dt_Customer"].str.extract(r'(\d{4})$').value_counts()
    complain = df["Complain"].value_counts()
    return complain, customer_dt_year, education, kidhome, martial, teenhome


@app.cell
def _(education):
    edu_values = education.values.tolist()
    edu_indexes = education.index.tolist()
    return edu_indexes, edu_values


@app.cell
def _(martial):
    mar_values = martial.values.tolist()
    mar_indexes = martial.index.tolist()

    mar_values[1], mar_values[-1] = mar_values[-1], mar_values[1]
    mar_indexes[1], mar_indexes[-1] = mar_indexes[-1], mar_indexes[1]

    mar_values[3], mar_values[-2] = mar_values[-2], mar_values[3]
    mar_indexes[3], mar_indexes[-2] = mar_indexes[-2], mar_indexes[3]

    mar_values[-3], mar_values[-2] = mar_values[-2], mar_values[-3]
    mar_indexes[-3], mar_indexes[-2] = mar_indexes[-2], mar_indexes[-3]
    return mar_indexes, mar_values


@app.cell
def _(kidhome):
    kid_values = kidhome.values.tolist()
    kid_indexes = kidhome.index.tolist()
    return kid_indexes, kid_values


@app.cell
def _(teenhome):
    teen_values = teenhome.values.tolist()
    teen_indexes = teenhome.index.tolist()
    return teen_indexes, teen_values


@app.cell
def _(customer_dt_year):
    customer_values = customer_dt_year.values.tolist()
    customer_indexes = [int(dt[0]) for dt in customer_dt_year.index.tolist()]
    return customer_indexes, customer_values


@app.cell
def _(complain):
    comp_values = complain.values.tolist()
    comp_indexes = complain.index.tolist()
    return comp_indexes, comp_values


@app.cell
def _(
    comp_indexes,
    comp_values,
    customer_indexes,
    customer_values,
    edu_indexes,
    edu_values,
    generate_pie,
    kid_indexes,
    kid_values,
    mar_indexes,
    mar_values,
    plt,
    teen_indexes,
    teen_values,
):

    plt.figure(figsize=(10, 14))

    generate_pie(edu_values, edu_indexes, [0.05, 0.05, 0.05, 0.1, 0.1], 1, "lower right", (0.15, 0.7), "Distribution by Education")

    generate_pie(mar_values, mar_indexes, [0.05, 0.05, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2], 2, "lower left", (0.9, 0.5), "Distribution by Martial Status")

    generate_pie(kid_values, kid_indexes, [0.05, 0.05, 0.1], 3, "lower right", (0.1, 0.7), "Distribution by Kidhome")

    generate_pie(teen_values, teen_indexes, [0.05, 0.05, 0.1], 4, "lower left", (0.9, 0.7), "Distribution by Teenhome")

    generate_pie(customer_values, customer_indexes, [0.05, 0.05, 0.1], 5, "lower right", (0.15, 0.7), "Distribution by Dt_Customer")

    generate_pie(comp_values, comp_indexes, [0.05, 0.05], 6, "lower left", (0.9, 0.7), "Distribution by Complain")

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `Most customers have a higher education, and there are few customers with basic education.`

    `The small YOLO and Absurd classes can be removed as they are not informative, and Alone can be merged with the Single class.`

    `58% of customers do not have young children, and 52% of customers do not have teenagers.`

    `There are very few complaints from customers.`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###Amount spent by product categories over the past 2 years
    """)
    return


@app.cell
def _(generate_graphs):

    columns_to_plot = ['MntWines', 'MntFruits',
                       'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
                       'MntGoldProds']

    generate_graphs(columns_to_plot)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `There is a logical decrease in the number of customers as the amount of money spent increases.`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###Dependence of sales on income
    """)
    return


@app.cell
def _(df, pd, plt, sns):
    data = pd.melt(df,
                    id_vars=['Income'],
                    value_vars=['NumDealsPurchases', 'NumWebPurchases',
           'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth'],
                    var_name="sales_type",
                    value_name="sales_volume")

    sc = sns.relplot(data=data,
                x="Income",
                y="sales_volume",
                col="sales_type",
                kind="scatter",
                hue="sales_type",
                height=4,
                aspect=0.9,
                col_wrap=3,
                palette="bright")

    sc.fig.subplots_adjust(hspace=0.15)
    plt.suptitle("Dependence of sales on income", y=1.05)
    plt.show()
    return


@app.cell
def _(df):
    df["TotalPurchases"] = df["NumCatalogPurchases"] + df["NumStorePurchases"] + df["NumWebPurchases"]
    df["WebSharePurchases"] = df["NumWebPurchases"] / df["TotalPurchases"]
    df["StoreSharePurchases"] = df["NumStorePurchases"] / df["TotalPurchases"]
    df["CatalogSharePurchases"] = df["NumCatalogPurchases"] / df["TotalPurchases"]

    df["WebSharePurchases"] = df["WebSharePurchases"].fillna(0)
    df["StoreSharePurchases"] = df["StoreSharePurchases"].fillna(0)
    df["CatalogSharePurchases"] = df["CatalogSharePurchases"].fillna(0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `Fixed a possible division by 0 by filling the resulting NaN values with zeros.`
    """)
    return


@app.cell
def _(df, pd, plt, sns):
    data_s = pd.melt(df,
                    id_vars=['Income'],
                    value_vars=['WebSharePurchases', 'StoreSharePurchases', 'CatalogSharePurchases'],
                    var_name="sales_type",
                    value_name="sales_share")

    scs = sns.relplot(data=data_s,
                x="Income",
                y="sales_share",
                col="sales_type",
                kind="scatter",
                hue="sales_type",
                height=4,
                aspect=0.9,
                col_wrap=3,
                palette="bright")

    scs.fig.subplots_adjust(hspace=0.15)
    plt.suptitle("Dependence of sales share on income", y=1.05)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `There is no specific relationship between places of purchase and Income. There is a relationship between the number of discounted items and Income, with richer individuals purchasing fewer discounted items. Also, poorer people visit the Web slightly more often per month than richer people. Statistics also show that people are more likely to buy online than in physical stores.`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Dependence of place on children
    """)
    return


@app.cell
def _(df):
    df["Children"] = df["Kidhome"] + df["Teenhome"]
    return


@app.cell
def _(df, pd, plt, sns):
    data_c = pd.melt(df,
                    id_vars=['Children'],
                    value_vars=['WebSharePurchases', 'StoreSharePurchases', 'CatalogSharePurchases'],
                    var_name="sales_type",
                    value_name="sales_share")

    scc = sns.relplot(data=data_c,
                y="Children",
                x="sales_share",
                col="sales_type",
                kind="scatter",
                hue="sales_type",
                height=4,
                aspect=0.9,
                col_wrap=3,
                palette="bright")

    scc.fig.subplots_adjust(hspace=0.15)
    plt.suptitle("Dependence of place on children", y=1.05)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `The number of children has affected the share of purchases in Catalog, but has had almost no effect on the share of purchases in Web and Store.`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###Dependence of TotalPurchases and Income on Children
    """)
    return


@app.cell
def _(df, plt, sns):
    plt.figure(figsize=(10, 8))

    plt.subplot(1, 2, 1)
    sns.boxenplot(x="Children", y="TotalPurchases", data=df)

    plt.subplot(1, 2, 2)
    sns.boxenplot(x="Children", y="Income", data=df)

    plt.suptitle("Dependence of TotalPurchases and Income on Children")

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `The number of children affects the number of purchases and the amount spent, resulting in a chain: more children -> lower income -> lower purchase in stores.`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###Distribution by campaigns
    """)
    return


@app.cell
def _(df):
    name_campaigns = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4',
           'AcceptedCmp5', 'Response']
    count_customers = [df[df[name_col] == 1].shape[0] for name_col in name_campaigns]
    return count_customers, name_campaigns


@app.cell
def _(count_customers, name_campaigns, plt, sns):
    plt.figure(figsize=(10, 6))

    ax = sns.barplot(
        x=name_campaigns,
        y=count_customers,
        fill=False,
        palette="Set1",
        hue=name_campaigns,
        linewidth=10,
        width=0.6,
    )

    for container in ax.containers:
        for patch in container.patches:
            if patch.get_height() > 0:
                x_pos = patch.get_x() + patch.get_width() / 2
                y_pos = patch.get_height() + 10          
                ax.text(
                    x_pos, y_pos, 
                    f'{int(patch.get_height())}', 
                    ha='center', va='center', 
                    fontsize=10, color='black'
                )

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `After analyzing all the campaigns, we can say that the second campaign failed, while the last one was quite successful.`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###Correlation
    """)
    return


@app.cell
def _(df):
    df_copy = df.drop(['ID', 'Kidhome','Teenhome',
           'NumCatalogPurchases', 'NumStorePurchases', 'NumWebPurchases',
           'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1',
           'AcceptedCmp2', 'Response', 'Complain', 'Z_CostContact', 'Z_Revenue'], axis=1)
    return (df_copy,)


@app.cell
def _(df_copy, plt, sns):
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_copy.corr(numeric_only=True), annot=True, cmap="coolwarm")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `Based on the correlation, we can say that all the connections are logical, and there are no large, unreasonable values. There is a direct relationship between TotalPurchase and meat, wine, and Income, which is logical.`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Multicolleniarity
    """)
    return


@app.cell
def _(pd, sm, variance_inflation_factor):
    def generate_vif(df):
        df_copy_mult = sm.add_constant(df)

        vif_df = pd.DataFrame()
        vif_df["feature"] = df_copy_mult.columns
        vif_df["VIF"] = [variance_inflation_factor(df_copy_mult.values, i)
                         for i in range(df_copy_mult.shape[1])]

        return vif_df.sort_values("VIF", ascending=False)

    return (generate_vif,)


@app.cell
def _(df_copy, generate_vif):
    df_copy_n = df_copy.select_dtypes(["number"])
    df_copy_n["Income"] = df_copy_n["Income"].fillna(0)  # replaced NaN with stubs to test multicollinearity

    generate_vif(df_copy_n)
    return (df_copy_n,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `It can be seen that CatalogSharePurchases, StoreSharePurchases, and WebSharePurchases are related because they are linearly dependent and add up to 1. I delete CatalogSharePurchases.`
    """)
    return


@app.cell
def _(df_copy_n, generate_vif):
    df_copy_wi = df_copy_n.drop(["CatalogSharePurchases"], axis=1)

    generate_vif(df_copy_wi)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `Now multicollinearity is normal.`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Feature engineering
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##New features
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - Age = 2014 - Year_Birth
    - DaysInCompany = 29.06.2014 - Dt_Customer
    - Children = Kidhome + Teenhome
    - TotalPurchases = NumWebPurchases + NumCatalogPurchases +  NumStorePurchases
    - TotalMnt = MntWines + MntFruits + MntMeatProducts + MntFishProducts + MntSweetProducts + MntGoldProds
    - TotalAcceptedCmp = AcceptedCmp1 + AcceptedCmp2 + AcceptedCmp3 + AcceptedCmp4 + AcceptedCmp5 + Response
    - WebSharePurchases = NumWebPurchases / TotalPurchases
    - StoreSharePurchases = NumStorePurchases / TotalPurchases
    - CatalogSharePurchases = NumCatalogPurchases / TotalPurchases
    """)
    return


@app.cell
def _(datetime, df, pd):
    df["Age"] = 2014 - df["Year_Birth"]

    df["DaysInCompany"] = (datetime(2014, 6, 29) - pd.to_datetime(df["Dt_Customer"], format="%d-%m-%Y")).dt.days.astype(int)

    df["TotalAcceptedCmp"] = df["AcceptedCmp1"] + df["AcceptedCmp2"] + df["AcceptedCmp3"] + df["AcceptedCmp4"] + df["AcceptedCmp5"] + df["Response"]

    df["TotalMnt"] = df["MntFishProducts"] + df["MntFruits"] + df["MntGoldProds"] + df["MntMeatProducts"] + df["MntSweetProducts"] + df["MntWines"] 
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Delete extra features
    """)
    return


@app.cell
def _(df):
    df_result = df.drop(['ID', 'Kidhome', 'Year_Birth',
           'Teenhome', 'MntWines', 'MntFruits',
           'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
           'MntGoldProds', 'NumWebPurchases',
           'NumCatalogPurchases', 'NumStorePurchases',
           'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1',
           'AcceptedCmp2', 'Z_CostContact', 'Z_Revenue', 'Response', 'CatalogSharePurchases', 'Dt_Customer'], axis=1)
    return (df_result,)


@app.cell
def _(df_result):
    df_result.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Processing Outliers
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###Income
    """)
    return


@app.cell
def _(df_result):
    df_result['Income'].describe(percentiles=[0.9, 0.95, 0.99])
    return


@app.cell
def _(df_result):
    df_result[df_result["Income"] == df_result["Income"].max()]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `According to the data, it doesn't look like an outlier. Maybe the customer has a really big income. However, it will be very difficult for the model to work with such a value, so it is better to remove.`
    """)
    return


@app.cell
def _(df_result):
    df_result_cleaned_temp = df_result[df_result["Income"] != df_result["Income"].max()]
    return (df_result_cleaned_temp,)


@app.cell
def _(df_result_cleaned_temp):
    df_result_cleaned_temp[df_result_cleaned_temp["Income"] < df_result_cleaned_temp["TotalMnt"]]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `There are no people who have spent more than their income.`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###Age
    """)
    return


@app.cell
def _(df_result_cleaned_temp):
    df_result_cleaned_temp[df_result_cleaned_temp["Age"] >= 95]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `People who are over 100 years old are most likely outliers.
    """)
    return


@app.cell
def _(df_result_cleaned_temp):
    df_result_cleaned_temp1 = df_result_cleaned_temp[df_result_cleaned_temp["Age"] < 100]
    return (df_result_cleaned_temp1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###Martial Status
    """)
    return


@app.cell
def _(df_result_cleaned_temp1):
    df_result_cleaned_temp2 = df_result_cleaned_temp1[~df_result_cleaned_temp1["Marital_Status"].isin(["YOLO", "Absurd"])]
    return (df_result_cleaned_temp2,)


@app.cell
def _(df_result_cleaned_temp2):
    df_result_cleaned_temp2["Marital_Status"] = df_result_cleaned_temp2["Marital_Status"].replace("Alone", "Single")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `Removed the outliers and merged Alone with Single.`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###Describe data
    """)
    return


@app.cell
def _(df_result_cleaned_temp2):
    df_result_cleaned_temp2.describe(percentiles=[0.8, 0.9, 0.95, 0.99])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `There are no more outliers.`
    """)
    return


@app.cell
def _(df_result_cleaned_temp2):
    df_result_cleaned = df_result_cleaned_temp2
    return (df_result_cleaned,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preprocessing data
    """)
    return


@app.cell
def _():
    from sklearn.pipeline import Pipeline
    from sklearn.impute import KNNImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
    from sklearn.compose import ColumnTransformer

    return (
        ColumnTransformer,
        FunctionTransformer,
        KNNImputer,
        OneHotEncoder,
        Pipeline,
        StandardScaler,
    )


@app.cell
def _(df_result_cleaned):
    numeric_features = df_result_cleaned.select_dtypes("number").drop(["Complain"], axis=1).columns

    category_features = df_result_cleaned.select_dtypes("object").columns

    bool_feature = ["Complain"]
    return bool_feature, category_features, numeric_features


@app.cell
def _(
    FunctionTransformer,
    KNNImputer,
    OneHotEncoder,
    Pipeline,
    StandardScaler,
):
    transformer_nums = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("imputer", KNNImputer(n_neighbors=5))
    ])

    transformer_cat = Pipeline(steps=[
        ("onehot", OneHotEncoder(drop="first"))
    ])

    transformer_bools = FunctionTransformer(lambda x: x, feature_names_out='one-to-one')
    return transformer_bools, transformer_cat, transformer_nums


@app.cell
def _(
    ColumnTransformer,
    bool_feature,
    category_features,
    numeric_features,
    transformer_bools,
    transformer_cat,
    transformer_nums,
):
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", transformer_nums, numeric_features),
            ("caregory", transformer_cat, category_features),
            ("bool", transformer_bools, bool_feature)
        ]
    )
    return (preprocessor,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Models
    """)
    return


@app.cell
def _():
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift, AffinityPropagation, Birch, SpectralClustering
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA

    return (
        AffinityPropagation,
        AgglomerativeClustering,
        Birch,
        DBSCAN,
        GaussianMixture,
        KMeans,
        MeanShift,
        PCA,
        SpectralClustering,
        silhouette_score,
    )


@app.cell
def _(df_result_cleaned):
    df_model = df_result_cleaned.copy()
    return (df_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##KMeans
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###Selecting the number of clusters
    """)
    return


@app.cell
def _(KMeans, df_model, plt, preprocessor, silhouette_score):
    X_processed = preprocessor.fit_transform(df_model)

    wcss = []
    silhouette_scores_kmeans = []
    max_k = 20

    for k in range(2, max_k + 1):
        kmeans_temp = KMeans(n_clusters=k, random_state=42)
        labels = kmeans_temp.fit_predict(X_processed)
        wcss.append(kmeans_temp.inertia_)
        score = silhouette_score(X_processed, labels)
        silhouette_scores_kmeans.append(score)

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(2, max_k + 1), wcss, marker='o', linestyle='--', color='b')
    plt.title("Elbow Method")
    plt.xlabel("Number of clusters")
    plt.ylabel("WCSS")
    plt.xticks(range(2, max_k + 1))
    plt.grid(True, alpha=0.5)

    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_k + 1), silhouette_scores_kmeans, marker='o', linestyle='-', color='green')
    plt.title("Silhouette Score")
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette Score")
    plt.xticks(range(2, max_k + 1))
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()
    return X_processed, max_k, silhouette_scores_kmeans


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `I select 3 clusters based on metrics.`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###Fit
    """)
    return


@app.cell
def _(KMeans, Pipeline, preprocessor):
    pipeline_kmeans = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("kmeans", KMeans(n_clusters=3, random_state=42))
    ])
    return (pipeline_kmeans,)


@app.cell
def _(df_model, pipeline_kmeans):
    pipeline_kmeans.fit(df_model)
    return


@app.cell
def _(df_result_cleaned, pipeline_kmeans):
    cluster_labels = pipeline_kmeans.named_steps["kmeans"].labels_
    df_result_cleaned["Cluster_KMeans"] = cluster_labels
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###Principal Component Analysis
    """)
    return


@app.cell
def _(pd, plt, sns):
    def generate_pca(df, hue, title="2D Principal Component Analysis (PCA)"):

        sns.scatterplot(
            x="PCA1", y="PCA2", 
            hue=hue, 
            data=df, 
            palette="bright", 
            s=60, alpha=0.8
        )
        plt.title(title)

    def generate_heatmap(preprocessor, pca):
        pca_loadings = pd.DataFrame(
            pca.components_, 
            columns=preprocessor.get_feature_names_out(),
            index=['PCA1', 'PCA2']
        ).T

        plt.figure(figsize=(10, 6))
        sns.heatmap(pca_loadings, cmap='coolwarm', center=0, annot=True, fmt='.3f')
        plt.title('PCA Loadings')
        plt.ylabel('Principal Components')
        plt.tight_layout()
        plt.show()

    return generate_heatmap, generate_pca


@app.cell
def _(PCA, X_processed, df_result_cleaned):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_processed)

    df_result_cleaned["PCA1"] = X_pca[:, 0]
    df_result_cleaned["PCA2"] = X_pca[:, 1]
    return (pca,)


@app.cell
def _(df_result_cleaned, generate_pca, plt):
    plt.figure(figsize=(10, 6))
    generate_pca(df_result_cleaned, hue="Cluster_KMeans")
    plt.grid(True, alpha=0.3)
    plt.show()
    return


@app.cell
def _(generate_heatmap, pca, preprocessor):
    generate_heatmap(preprocessor, pca)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###Cluster analytics
    """)
    return


@app.cell
def _(df_model, df_result_cleaned):
    cluster_profiles = df_result_cleaned.groupby("Cluster_KMeans")[
        df_model.select_dtypes("number").columns
    ].mean().round(2)

    cluster_profiles
    return


@app.cell
def _(df_result_cleaned, math, plt, sns):
    target_column = "TotalMnt"
    comparable_columns = ['Income', 'Children', 'TotalPurchases', 'DaysInCompany', 'WebSharePurchases', 'StoreSharePurchases']

    plt.figure(figsize=(16, 10))

    num_col_in_row = 3
    num_row = math.ceil(len(comparable_columns) / num_col_in_row)

    for i, col in enumerate(comparable_columns):
        plt.subplot(num_row, num_col_in_row, i+1)
        sns.scatterplot(
            y=col, 
            x=target_column, 
            hue="Cluster_KMeans", 
            data=df_result_cleaned, 
            palette="bright", 
            alpha=0.7
        )
        plt.title(f"Dependence: {col} vs {target_column}", fontdict={"fontsize": 15})
        plt.ylabel(col, fontdict={"fontsize": 14})
        plt.xlabel(target_column, fontdict={"fontsize": 14})

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `Conclusions based on graphs:`
    - `class 0 - people with minimal Income, who buy the least and most often in Store, and most of them have children`
    - `class 1 - people with maximal Income, who spend the most and buy more often, prefer Store more often, almost no children`
    - `class 2 - people with average Income, family people with children, who buy more often in Web`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##AgglomerativeClustering
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###Selecting the number of clusters
    """)
    return


@app.cell
def _(
    AgglomerativeClustering,
    X_processed,
    max_k,
    plt,
    silhouette_score,
    silhouette_scores_kmeans,
):

    silhouette_scores_agl = []

    for k_agl in range(2, max_k + 1):
        agl_temp = AgglomerativeClustering(n_clusters=k_agl)
        labels_agl = agl_temp.fit_predict(X_processed)
        score_agl = silhouette_score(X_processed, labels_agl)
        silhouette_scores_agl.append(score_agl)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(2, max_k + 1), silhouette_scores_agl, marker='o', linestyle='-', color='orange')
    plt.title("Silhouette Score Agglomerative Clustering")
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette Score")
    plt.xticks(range(2, max_k + 1))
    plt.grid(True, alpha=0.5)

    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_k + 1), silhouette_scores_kmeans, marker='o', linestyle='-', color='green')
    plt.title("Silhouette Score KMeans")
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette Score")
    plt.xticks(range(2, max_k + 1))
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `The Silhouette Score of the two methods differs by about ~0.3, but also identifies three clusters (we do not take 2).`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###Fit
    """)
    return


@app.cell
def _(AgglomerativeClustering, Pipeline, preprocessor):
    pipeline_agl= Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("aglomerative", AgglomerativeClustering(n_clusters=3))
    ])
    return (pipeline_agl,)


@app.cell
def _(df_model, pipeline_agl):
    pipeline_agl.fit(df_model)
    return


@app.cell
def _(df_result_cleaned, pipeline_agl):
    cluster_labels_agl = pipeline_agl.named_steps["aglomerative"].labels_
    df_result_cleaned["Cluster_Agglomerative"] = cluster_labels_agl
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###Principal Component Analysis
    """)
    return


@app.cell
def _(df_result_cleaned, generate_pca, plt):
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    generate_pca(df_result_cleaned, hue="Cluster_Agglomerative", title="Agglomerative Clustering PCA")
    plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    generate_pca(df_result_cleaned, hue="Cluster_KMeans", title="KMeans PCA")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(df_model, df_result_cleaned):
    cluster_profiles_agl = df_result_cleaned.groupby("Cluster_Agglomerative")[
        df_model.select_dtypes("number").columns
    ].mean().round(2)

    cluster_profiles_agl
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `The agglomerative algorithm also produced 3 similar classes, but unlike KMeans, the classes are more mixed together. It can also be seen that the numbers of clusters 1 and 0 have changed.`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Other models
    """)
    return


@app.cell
def _(DBSCAN, X_processed, df_result_cleaned):
    dbscan = DBSCAN(eps=2.0, min_samples=10)
    labels_dbscan = dbscan.fit_predict(X_processed)

    df_result_cleaned["Cluster_DBSCAN"] = labels_dbscan
    return


@app.cell
def _(MeanShift, X_processed, df_result_cleaned):
    shift = MeanShift()
    labels_shift = shift.fit_predict(X_processed)

    df_result_cleaned["Cluster_Shift"] = labels_shift
    return


@app.cell
def _(AffinityPropagation, X_processed, df_result_cleaned):
    affinity = AffinityPropagation(random_state=42)
    labels_affinity = affinity.fit_predict(X_processed)

    df_result_cleaned["Cluster_Affinity"] = labels_affinity
    return


@app.cell
def _(Birch, X_processed, df_result_cleaned):
    birch = Birch(n_clusters=3)
    labels_birch= birch.fit_predict(X_processed)

    df_result_cleaned["Cluster_Birch"] = labels_birch
    return


@app.cell
def _(GaussianMixture, X_processed, df_result_cleaned):
    mixture = GaussianMixture(n_components=3, random_state=42)
    labels_mixture= mixture.fit_predict(X_processed)

    df_result_cleaned["Cluster_GMixture"] = labels_mixture
    return


@app.cell
def _(SpectralClustering, X_processed, df_result_cleaned):
    spectral = SpectralClustering(n_clusters=3, n_components=2, affinity='nearest_neighbors', n_neighbors=1500)
    labels_spectral= spectral.fit_predict(X_processed)

    df_result_cleaned["Cluster_Spectral"] = labels_spectral
    return


@app.cell
def _(df_result_cleaned, plt, sns):
    num_rows = 2
    num_columns = 4

    fig, axes = plt.subplots(num_rows, num_columns, figsize=(18, 10))

    algorithms = [
        ('K-Means', 'Cluster_KMeans'), 
        ('Agglomerative', 'Cluster_Agglomerative'), 
        ('Birch', 'Cluster_Birch'),
        ('SpectralClustering', 'Cluster_Spectral'),
        ('DBSCAN', 'Cluster_DBSCAN'),
        ('MeanShift', 'Cluster_Shift'),
        ('AffinityPropagation', 'Cluster_Affinity'),
        ('GaussianMixture', 'Cluster_GMixture'),
    ]

    axes_flat = axes.flatten()

    for ind, (title, cluster_col) in enumerate(algorithms):

        ax_ = axes_flat[ind]

        sns.scatterplot(
            data=df_result_cleaned,
            x='PCA1',
            y='PCA2',
            hue=cluster_col,
            palette='bright',
            ax=ax_,
            s=30,
            alpha=0.7
        )

        ax_.set_title(title, fontsize=16, pad=15)

        ax_.legend_.remove() if ax_.legend_ else None

        ax_.set_xticks([])
        ax_.set_yticks([])
        ax_.set_xlabel('')
        ax_.set_ylabel('')

        for spine in ax_.spines.values():
            spine.set_color('#CCCCCC')

    plt.suptitle("Comparison of clustering algorithms in PCA space", fontsize=20, y=1.05)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Conclusion
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `To summarize, this project encompassed an end-to-end customer personality analysis, including in-depth EDA, feature engineering, and model training. By comparing multiple clustering algorithms, I found that hierarchical and centroid-based methods performed best for this specific distribution.`

    `Ultimately, the customer base was segmented into three distinct and actionable profiles: Wealthy Customers, Family-Oriented Customers, and Low-Income Customers. This segmentation empowers the business to shift from a mass-marketing approach to targeted strategies, allowing for personalized promotional campaigns and tailored services that resonate with the specific buying habits of each group.`
    """)
    return


if __name__ == "__main__":
    app.run()
