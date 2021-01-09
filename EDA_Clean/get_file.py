import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

from pyspark.sql import SparkSession, functions, types
spark = SparkSession.builder.appName('elevator anom').getOrCreate()
assert spark.version >= '2.4' # make sure we have Spark 2.4+
spark.sparkContext.setLogLevel('WARN')
sc = spark.sparkContext

s_schema = types.StructType([
    types.StructField('datetime', types.DateType()),
    types.StructField('x', types.FloatType()),
    types.StructField('y', types.FloatType()),
    types.StructField('z', types.FloatType())
])

def main(inputs, output):
    df = spark.read.csv(inputs, schema=s_schema)
    df.createOrReplaceTempView("df_view")
    df = spark.sql("SELECT * FROM df_view WHERE df_view.datetime >= '2018-07-09 12:00:00' AND df_view.datetime <= '2018-08-09 12:00:00'") #Can change this to get smaller parts of the data - this section is ~67 million rows
    df.write.csv(output)

if __name__ == '__main__':
    inputs = sys.argv[1]
    output = sys.argv[2]
    main(inputs, output)