import polars as pl
import polars.selectors as cs

# improve print outputs from polars
pl.Config.set_tbl_rows(30)
pl.Config.set_tbl_width_chars(3000)
pl.Config.set_tbl_cols(-1)


# read in file
raw_train = pl.read_csv('data/train.csv', null_values=["",'-','–'])
raw_test = pl.read_csv('data/test.csv')

# preview file
raw_train.head()
raw_train.shape
raw_train.glimpse()

# check the number of null values
raw_train.select(pl.all().null_count()).unpivot().with_columns(pct_missing = (pl.col('value') / raw_train.shape[0]).round(2))

#### FEATURE EXTRACTION - engine
# this column has some good data that we can extract
def extract_displacement(df):
    ans = df.with_columns(displacement = pl.col('engine').str.extract(r"([0-9]{1}.[0-9]{1,2}L)|([0-9]{1}.[0-9]{1,2} L)",0).str.replace("L","").str.replace(" ", "").alias('displacement'))

    ans2 = ans \
        .with_columns(pl.when(pl.col('displacement').is_null()) \
                      .then(pl.col('engine').str.extract(r"[0-9]{1}.[0-9]{1} Liter", 0).str.replace("Liter", "")) \
                      .otherwise(pl.col('displacement')).alias('displacement')) \
        .with_columns(displacement = pl.col('displacement').cast(pl.Float64))

    return(ans2)

def extract_hp(df):
    ans = df.with_columns(pl.col('engine').str.extract(r"[0-9]\d{1,4}.[0-9]{1}HP",0).str.replace("HP","").alias('hp')).with_columns(hp = pl.col('hp').cast(pl.Float64))

    return(ans)

def extract_cyl(df):
    ans = df.with_columns(pl.col('engine').str.extract(r"([0-9]\d{0,2} Cylinder) | V[0-9]\d{0,2}|I3|I4|I6|H4|H6",0).str.replace(" Cylinder","").str.replace("V","").str.replace(" ","").str.replace("I","").str.replace("H","").alias('cylinders'))

    ans2 = ans.with_columns(pl.when((pl.col('cylinders').is_null()) & (pl.col('engine').str.contains("(?i)electric|(?i)dual|(?i)battery"))).then(pl.lit(0)).otherwise(pl.col('cylinders')).alias('cylinders'))

    ans3 = ans2.with_columns(cylinders = pl.col('cylinders').cast(pl.Float64))

    return(ans3)


cln_train = extract_displacement(df = raw_train)
cln_train = extract_hp(df = cln_train)
cln_train = extract_cyl(df = cln_train)

cln_train.head()
cln_train.describe()

cln_train.select('engine','hp').filter(pl.col('hp').is_null())

# ENGINE STUFF
# let's try to split engine with displacement
tmp = raw_train.select('engine')

# extract displacement
tmp_disp = tmp.with_columns(displacement = pl.col('engine').str.extract(r"([0-9]{1}.[0-9]{1,2}L)|([0-9]{1}.[0-9]{1,2} L)",0).str.replace("L","").str.replace(" ", "").alias('displacement'))

tmp_disp.filter(pl.col('displacement').is_null())

tmp_disp = tmp_disp \
    .with_columns(pl.when(pl.col('displacement').is_null()) \
                  .then(pl.col('engine').str.extract(r"[0-9]{1}.[0-9]{1} Liter", 0).str.replace("Liter", "")) \
                  .otherwise(pl.col('displacement')).alias('displacement'))

tmp_disp = tmp_disp.with_columns(pl.when((pl.col('displacement').is_null() & pl.col('engine').str.contains("(?i)electric|(?i)battery|(?i)dual"))).then(pl.lit(0)).otherwise(pl.col('displacement')).alias('displacement'))

tmp_disp.filter(pl.col('displacement').is_null())
tmp_disp.filter(pl.col('displacement').is_null()).glimpse()
tmp_disp.filter(pl.col('engine').is_null())
# tmp_disp.filter(pl.col('engine') == '–')

tmp_disp.with_columns(pl.col('displacement').cast(pl.Float64)).filter(pl.col('displacement').is_null())

# raw_train.filter(pl.col("engine") == '–')

tmp_disp.filter((pl.col('displacement').is_null()) & (pl.col('engine').str.contains('Intercooled Turbo'))).head().select('engine').glimpse()

tmp_disp.filter((pl.col('displacement').is_null()) & (pl.col('engine').str.contains("L/[0-9]{1,3}"))).with_columns(pl.col('engine').str.replace(r"L/[0-9]{1,3}", 'L').alias('disp2')).glimpse()


# extract HP
tmp.with_columns(pl.col('engine').str.extract(r"[0-9]\d{1,4}.[0-9]{1}HP",0).str.replace("HP","").alias('hp')).head(20)
tmp.with_columns(pl.col('engine').str.extract(r"[0-9]\d{1,4}.[0-9]{1}HP",0).str.replace("HP","").alias('hp')).filter(pl.col('hp').is_null())

tmp_hp = tmp.with_columns(pl.col('engine').str.extract(r"[0-9]\d{1,4}.[0-9]{1}HP",0).str.replace("HP","").alias('hp'))



# extract cylinder
tmp.with_columns(pl.col('engine').str.extract(r"([0-9]\d{0,2} Cylinder) | V[0-9]\d{0,2}|I4|I6|H4|H6",0).str.replace(" Cylinder","").str.replace("V","").str.replace(" ","").alias('cylinders')).head(20)

tmp_cyl = tmp.with_columns(pl.col('engine').str.extract(r"([0-9]\d{0,2} Cylinder) | V[0-9]\d{0,2}|I3|I4|I6|H4|H6",0).str.replace(" Cylinder","").str.replace("V","").str.replace(" ","").str.replace("I","").str.replace("H","").alias('cylinders'))

tmp_cyl = tmp_cyl.with_columns(pl.when((pl.col('cylinders').is_null()) & (pl.col('engine').str.contains("(?i)electric|(?i)dual|(?i)battery"))).then(pl.lit(0)).otherwise(pl.col('cylinders')).alias('cylinders'))

tmp_cyl.with_columns(cylinders = pl.col('cylinders').cast(pl.Float64))

tmp_cyl.head()
tmp_cyl.describe()

tmp_cyl.filter(pl.col('cylinders').is_null())

tmp_cyl.filter(pl.col('cylinders').is_null()).glimpse()

tmp_cyl.filter(pl.col('engine').str.contains('Rotary')).glimpse()

# MODEL
raw_train.select('model').with_columns(pl.col('model').str.extract(r"(\bAMG C 63\b) |(^[A-Za-z0-9\-]{1,20})",0).alias('model2')).unique()
 
# number of brands
raw_train.select('brand').n_unique()
raw_train.select('model').n_unique()
raw_train.select('fuel_type').n_unique()
raw_train.select('accident').n_unique()
raw_train.select('clean_title').n_unique()