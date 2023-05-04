import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


x = [0,1,2]
y = [100,200,300]

# chart1
# plt.plot(x,y)
# plt.show()


housing = pd.DataFrame({"rooms":[1,1,2,2,2,3,3,3],
                        "price":[100,120,190,200,230,310,330,305]
                        })

# chart2
# plt.scatter(housing["rooms"],housing["price"])
# plt.show()


# chart3
# plt.plot(x,y)
# plt.title("title")
# plt.xlabel("X Label")
# plt.ylabel("Y Label")
# # to prevent output in the console add a ; after the last plt
# # command
# plt.show()


# chart4
# limit the plot axis
# plt.plot(x,y)
# # after the plt.plot command use xlim and/or ylim
# plt.xlim(0,2)
# plt.ylim(0,300)
# plt.title("title")
# plt.xlabel("X Label")
# plt.ylabel("Y Label")
# plt.show()


# chart5
# change plot color, marker, markersize,linestyle
plt.plot(x,y,color="#bc67c9",
        marker="*",
        markersize=15,
        linestyle="--"
        )
plt.xlim(0,2)
plt.ylim(0,300)
plt.title("title")
plt.xlabel("X Label")
plt.ylabel("Y Label")
plt.show()











