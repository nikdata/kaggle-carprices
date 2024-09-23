import polars as pl
import polars.selectors as cs
import matplotlib.pyplot as plt
import seaborn as sns
import plotly_express as px
import seaborn.objects as so

# improve print outputs from polars
pl.Config.set_tbl_rows(30)
pl.Config.set_tbl_width_chars(3000)
pl.Config.set_tbl_cols(-1)

# read in train & test file
raw_train = pl.read_csv('data/train.csv', null_values=["",'-','–'])
raw_test = pl.read_csv('data/test.csv', null_values = ['','-','–'])

# quick info on df
raw_train.shape # 188533 rows, 13 columns with response variable
raw_train.sample(n = 10)
raw_train.glimpse()
raw_train.describe()

raw_test.shape # 125690 rows, 12 columns not including response variable
raw_test.sample(n = 10)
raw_test.glimpse()
raw_test.describe()

# get skew & kurtosis numerical columns

raw_train \
    .select(cs.by_dtype(pl.Int64(), pl.Float64)) \
    .select(pl.exclude('model_year','id')) \
    .describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]) \
    .vstack(pl.DataFrame({'statistic': 'skew', 'milage': raw_train.select(pl.col('milage').skew()), 'price': raw_train.select(pl.col('price').skew())})) \
    .vstack(pl.DataFrame({'statistic': 'kurtosis', 'milage': raw_train.select(pl.col('milage').kurtosis(fisher=True)), 'price': raw_train.select(pl.col('price').kurtosis(fisher=True))}))


# find missing value count by column
raw_train \
    .select(pl.all().null_count()) \
    .unpivot(value_name = 'missing') \
    .with_columns(pct = (pl.col('missing') / raw_train.shape[0]).round(3))\
    .sort(by = 'pct', descending = True)

# find total number of models by brand - train/test
raw_train.group_by(['brand']).agg(models = pl.col('model').len()).sort(by = ['models'], descending=True).with_columns(pct = (pl.col('models') / (pl.col('models').sum())).round(3))
raw_test.group_by(['brand']).agg(models = pl.col('model').len()).sort(by = ['models'], descending=True).with_columns(pct = (pl.col('models') / (pl.col('models').sum())).round(3))

# make sure brands found in train are in test as well
train_brands = raw_train.select('brand').unique()
test_brands = raw_test.select('brand').unique()

test_brands.filter(~pl.col('brand').is_in(train_brands.select('brand'))) # all brands in test are also in train

# get price stats by brand
raw_train.group_by(['brand']).agg(models = pl.col('model').len(), pct = ((pl.col('model').len() / raw_train.shape[0])* 100).round(2), min_price = pl.col('price').min(), median_price = pl.col('price').median(), avg_price = pl.col('price').mean(), max_price = pl.col('price').max(), sd_price = pl.col('price').std()).sort(by = 'models', descending=True)

# MODELS
raw_train.group_by(['model']).agg(models = pl.col('model').len(), pct = ((pl.col('model').len() / raw_train.shape[0])* 100).round(2), min_price = pl.col('price').min(), median_price = pl.col('price').median(), avg_price = pl.col('price').mean(), max_price = pl.col('price').max(), sd_price = pl.col('price').std()).sort(by = 'models', descending=True)
# there are almost 1900 unique models - too many

# model year
raw_train.group_by(['model_year']).agg(models = pl.col('model').len(), pct = ((pl.col('model').len() / raw_train.shape[0])* 100).round(2), min_price = pl.col('price').min(), median_price = pl.col('price').median(), avg_price = pl.col('price').mean(), max_price = pl.col('price').max(), sd_price = pl.col('price').std()).sort(by = 'models', descending=True)


# check distribution of mileage
fig = so.Plot(data = raw_train, x = 'milage') \
    .add(so.Bars(), so.Hist(bins = 20)) \
    .label(x = 'Mileage', y = 'Count',title = "Distribution of Mileage")
fig.show()

# check out some charts on price (the response variable)
fig = so.Plot(data = raw_train, x = 'price') \
    .add(so.Bars(), so.Hist(bins = 20)) \
    .label(x = 'Price', y = 'Count', title = 'Distribution of Price')

fig.show()

raw_train.head()
raw_train.select('price').describe(percentiles = [0.25, 0.50, 0.75, 0.90, 0.95, 0.99])

fig = so.Plot(data = raw_train.filter(pl.col('price') < 108000), x = 'price') \
    .add(so.Bars(), so.Hist(bins = 20)) \
    .label(x = 'Price', y = 'Count', title = 'Distribution of Price')

fig.show()




# check XY relationship between mileage and price

fig = so.Plot(data = raw_train, x = 'milage', y = 'price').add(so.Dots(color = 'g')).scale(x = 'log', y='log').label(x = 'Mileage', y = 'Price', title = 'Price vs. Mileage')
fig.show()

# let's explore fuel type

raw_train.group_by('fuel_type').agg(pl.col('fuel_type').len().alias('obs')).with_columns(pct = ((pl.col('obs')/pl.col('obs').sum())*100).round(2)).sort(by = 'pct', descending = True)

raw_test.group_by('fuel_type').agg(pl.col('fuel_type').len().alias('obs')).with_columns(pct = ((pl.col('obs')/pl.col('obs').sum())*100).round(2)).sort(by = 'pct', descending = True)

raw_train.filter(pl.col('fuel_type') == 'not supported').select('engine').glimpse()