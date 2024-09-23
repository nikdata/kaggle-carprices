# import libraries
import polars as pl
import polars.selectors as cs
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so


# improve print outputs from polars
pl.Config.set_tbl_rows(30)
pl.Config.set_tbl_width_chars(3000)
pl.Config.set_tbl_cols(-1)

# read in train & test file
raw_train = pl.read_csv('data/train.csv', null_values=["",'-','–'])
raw_test = pl.read_csv('data/test.csv', null_values = ['','-','–'])

# time to extract things
raw_train.select('engine').with_columns(hp = pl.col('engine').str.extract(r"(\d{1,4}\.?0?)HP"))
raw_train.select('engine').with_columns(displacement=pl.col('engine').str.extract(r"(\d+\.?\d*)\s*(L|Liter)"))
raw_train.select('engine').with_columns(cylinder = pl.col('engine').str.extract(r'(\d+)\s*Cylinder|V(\d+)|I(\d+)'))
raw_train.select('engine').with_columns(fuel = pl.col('engine').str.extract(r'(Gasoline|Diesel|Flex Fuel|Hydrogen|Electric|Hybrid)'))
raw_train.select('engine').with_columns(config = pl.col('engine').str.extract(r'(I\d+|V\d+|H\d+|Flat \d+|Straight \d+|Rotary)'))

def extract_features(df):
    ans = df \
        .with_columns(hp = pl.col('engine').str.extract(r"(\d{1,4}\.?0?)HP").cast(pl.Float64())) \
        .with_columns(displacement=pl.col('engine').str.extract(r"(\d+\.?\d*)\s*(L|Liter)").cast(pl.Float64())) \
        .with_columns(cylinder = pl.col('engine').str.extract(r'(\d+)\s*Cylinder|V(\d+)|I(\d+)').cast(pl.Float64())) \
        .with_columns(fuel = pl.col('engine').str.extract(r'(Gasoline|Diesel|Flex Fuel|Hydrogen|Electric|Hybrid)').str.to_lowercase()) \
        .with_columns(engine_config = pl.col('engine').str.extract(r'(I\d+|V\d+|H\d+|Flat \d+|Straight \d+|Rotary)'))

    return(ans)

raw_train_extract = extract_features(df = raw_train)
raw_test_extract = extract_features(df = raw_test)

raw_train_extract.glimpse()
raw_train_extract.select('fuel').unique()

# let's make a histogram for horsepower

fig = so.Plot(data = raw_train_extract, x = 'hp') \
    .add(so.Bars(), so.Hist(bins = 20)) \
    .label(x = 'hp', y = 'Count',title = "Distribution of HP")
fig.show()

fig = so.Plot(data = raw_train_extract, x = 'cylinder') \
    .add(so.Bars(), so.Hist(bins = 20)) \
    .label(x = 'cylinders', y = 'Count',title = "Distribution of Cylinders")
fig.show()

# hp vs price

fig = so.Plot(data = raw_train_extract.filter(pl.col('price') < 1000000), x = 'hp', y = 'price') \
    .add(so.Dots()) \
    .label(x = 'hp',y = 'price', title = 'Price vs. HP')
fig.show()