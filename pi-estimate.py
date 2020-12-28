import random
import pyspark

# This is a super simple example of how pyspark works to parallelize computation, local machine only
# The local[*] string is a special string denoting that you’re using a local cluster,
# which is another way of saying you’re running in single-machine mode.
# The * tells Spark to create as many worker threads as logical cores on your machine.
sc = pyspark.SparkContext('local[*]')

# Spark can also be used for compute-intensive tasks. This code estimates π by "throwing darts" at a circle.
# We pick random points in the unit square ((0, 0) to (1,1)) and see how many fall in the unit circle.
# The fraction should be π / 4, so we use this to get our estimate.
# https://spark.apache.org/examples.html

# The reason for self as the argument is to make it an instance method required by sc.parallelize().filter()
# Even though it's not really used.
# There doesn't seem to be a way to pass a static method to filter()
def inside(self):
    x, y = random.random(), random.random()
    return x*x + y*y < 1


NUM_SAMPLES = 100000000
count = 0

# This is the non-spark version of the calculation
for i in range(0, NUM_SAMPLES):
    if inside(None):
        count = count + 1

# spark version
# count = sc.parallelize(range(0, NUM_SAMPLES)) \
#              .filter(inside).count()


print("Pi is roughly %f" % (4.0 * count / NUM_SAMPLES))
