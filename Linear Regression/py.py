import random
import pandas as pd
import numpy as np
np.random.seed(1)
Size=50_000
data3=pd.DataFrame({
    "studytime" :np.random.randint(low=20.000000, high=80.000000,size=Size,),
    "Score" :np.random.randint(low=20.000000, high=80.000000,size=Size)
})
df=pd.DataFrame(data3)
df.to_csv("data.csv",index=False)
print(df)