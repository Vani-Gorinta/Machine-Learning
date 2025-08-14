import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport

# Load dataset
data = pd.read_csv(r"C:\Users\vanig\Downloads\archive (1)\Housing.csv")
print(data)

# Quick info
data.info()
data.head(10)

# Generate profile report
profile = ProfileReport(data)
profile.to_file("report.html")

from ydata_profiling import ProfileReport
prof = ProfileReport(data)
prof.to_file(output_file='EDA.html')
from IPython.core.display import display, HTML
display(HTML(prof.to_html()))