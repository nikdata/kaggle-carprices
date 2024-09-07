import polars as pl
import polars.selectors as cs

# improve print outputs from polars
pl.Config.set_tbl_rows(30)
pl.Config.set_tbl_width_chars(2000)
pl.Config.set_tbl_cols(-1)


# read in file
raw_train = pl.read_csv('data/train.csv')
raw_test = pl.read_csv('data/test.csv')

# preview file
raw_train.head()
raw_train.shape
raw_train.glimpse()

#### FEATURE EXTRACTION


# ENGINE STUFF
# let's try to split engine with displacement
tmp = raw_train.select('engine')

# extract displacement
tmp.with_columns(displacement = pl.col('engine').str.extract(r"[0-9]{1}.[0-9]{1}L",0).str.replace("L","").alias('displacement')).head(20)

# extract HP
tmp.with_columns(pl.col('engine').str.extract(r"[0-9]\d{1,4}.[0-9]{1}HP",0).str.replace("HP","").alias('hp')).head(20)

# extract cylinder
tmp.with_columns(pl.col('engine').str.extract(r"([0-9]\d{0,2} Cylinder) | V[0-9]\d{0,2}",0).str.replace(" Cylinder","").str.replace("V","").str.replace(" ","").alias('cylinders')).head(20)

# MODEL
raw_train.select('model').with_columns(pl.col('model').str.extract(r"(\bAMG C 63\b) |(^[A-Za-z0-9\-]{1,20})",0).alias('model2')).unique()
 
# number of brands
raw_train.select('brand').n_unique()
raw_train.select('model').n_unique()
raw_train.select('fuel_type').n_unique()
raw_train.select('accident').n_unique()
raw_train.select('clean_title').n_unique()