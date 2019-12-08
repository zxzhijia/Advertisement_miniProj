#2 - Based on number of movies watched, agewise who is easy to please
import pandas as pd
##Age wise number of movies watched
Wholedata = pd.read_csv('/Users/apple/Desktop/ml-1m/MovieData1.csv', header=0)
Wholedata.head()
df2=pd.DataFrame(Wholedata.groupby(["Genre","Title","Age"]).size())
df2["Count"]=df2
del df2[0]
agegenre=df2
agegenre.reset_index(level=["Title","Genre","Age"], inplace=True)
agegenre1=agegenre[(agegenre["Age"]>=18) & (agegenre["Age"]<=24)]
agegenre2=agegenre[(agegenre["Age"]>=25) & (agegenre["Age"]<=34)]
agegenre3=agegenre[(agegenre["Age"]>=35) & (agegenre["Age"]<=44)]
agegenre4=agegenre[(agegenre["Age"]>=45) & (agegenre["Age"]<=49)]
agegenre5=agegenre[(agegenre["Age"]>=50) & (agegenre["Age"]<=55)]
agegenre6=agegenre[(agegenre["Age"]>=56)]
agegenre7=agegenre[(agegenre["Age"]<18)]
agegenre1=agegenre1.sort_values(by=["Count"],ascending=False)
agegenre2=agegenre2.sort_values(by=["Count"],ascending=False)
agegenre3=agegenre3.sort_values(by=["Count"],ascending=False)
agegenre4=agegenre4.sort_values(by=["Count"],ascending=False)
agegenre5=agegenre5.sort_values(by=["Count"],ascending=False)
agegenre6=agegenre6.sort_values(by=["Count"],ascending=False)
agegenre7=agegenre7.sort_values(by=["Count"],ascending=False)

#plot age wise number of movies watched
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
fontsize=15
plt.bar((1,2,3,4,5,6,7),(sum(agegenre1["Count"]),sum(agegenre2["Count"]),sum(agegenre3["Count"]),sum(agegenre4["Count"]),sum(agegenre5["Count"]),sum(agegenre6["Count"]),sum(agegenre7["Count"])), alpha=0.4,color='r')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.xticks((1.4,2.4,3.4,4.4,5.4,6.4,7.4),("18-24","25-34","35-44","45-49","50-55","56+","Under 18"))
plt.xlabel('Age Ranges',fontsize=15)
plt.ylabel('Number of Movies Rated',fontsize=15)
plt.title('Age wise Number of Movies Watched',fontsize=15)
plt.show()

##Age wise average rating given
Wholedata = pd.read_csv('C:/Users/Deepan Sanghavi/501/Case Study 2/Karan/MovieData1.csv', header=0)
Wholedata.head()
df2=pd.DataFrame(Wholedata.groupby(["Age"])["Rating"].mean())
df2