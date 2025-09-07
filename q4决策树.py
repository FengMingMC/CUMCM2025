import pydot
from mpmath import plot
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree

# from main import female_data_content
from q4清洗数据 import *

calculate_frame = pd.DataFrame({
    "WeekNum": df_girl["孕周"],
    "BMI": df_girl["孕妇BMI"],
    "Age": df_girl["年龄"],
    "Height": df_girl["身高"],
    "Weight": df_girl["体重"],
    "GC": df_girl["GC含量"],
    "OriginReadNumber": df_girl["原始读段数"],
    "ComparisonProportion": df_girl["在参考基因组上比对的比例"],
    "RepeatProportion": df_girl["重复读段的比例"],
    "OnlyComparisonProportion": df_girl["唯一比对的读段数"],
    "13_Z": df_girl["13号染色体的Z值"],
    "18_Z": df_girl["18号染色体的Z值"],
    "21_Z": df_girl["21号染色体的Z值"],
    "X_Z": df_girl["X染色体的Z值"],
    "X_Content": df_girl["X染色体浓度"],
    "13_GC": df_girl["13号染色体的GC含量"],
    "18_GC": df_girl["18号染色体的GC含量"],
    "21_GC": df_girl["21号染色体的GC含量"],
    "FilterProportion": df_girl["被过滤掉读段数的比例"],
    "Method": df_girl["Method"],
    "PregnancyTimes": df_girl['PregnancyTimes'],
    "GiveBirthTimes": df_girl["生产次数"],

    "TripleChromosome": df_girl["TripleChromosome"]
    # "X染色体浓度": df_girl["X染色体浓度"],

})

features = ["WeekNum",
            "BMI",
            "Age",
            "Height",
            "Weight",

            "GC",
            "OriginReadNumber",
            "ComparisonProportion",
            "RepeatProportion",
            "OnlyComparisonProportion",

            "13_Z",
            "18_Z",
            "21_Z",
            "X_Z",
            "X_Content",

            "13_GC",
            "18_GC",
            "21_GC",
            "FilterProportion",
            "Method",

            "PregnancyTimes",
            "GiveBirthTimes",]

target = "TripleChromosome"

# print(calculate_frame["TripleChromosome"].value_counts())
# actual_class_names = ["Health", "Unhealth"]
#
X = calculate_frame[features]
y = calculate_frame[target]

dt_classifier = DecisionTreeClassifier(
    random_state=712 ,
    criterion='gini',
    max_depth=5,
    min_samples_leaf=5,
    min_samples_split=2,
)
class_names = ["Health", "Unhealth"]
dt_classifier.fit(X, y)
y_pred = dt_classifier.predict(X)
y_pred_proba = dt_classifier.predict_proba(X)

accuracy = accuracy_score(y, y_pred)
print(f"\n模型准确率：{accuracy:4f}")
print("\n分类报告")
print(classification_report(y, y_pred))


plt.figure(figsize=[20,12])
plot_tree(
    dt_classifier,
    filled=True,
    feature_names=calculate_frame.columns,
    class_names=class_names,
    rounded=True,
    proportion=True,
    precision=3,
    # fontname="helvetica",
)
plt.title("Decision Tree Classifier")
plt.savefig("DecisionTreeClassifier.pdf")
plt.show()

fpr, tpr, thresholds = roc_curve(y, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=[20,12])
plt.plot(fpr, tpr, label="AUC = %0.2f" % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.title('ROC曲线')
plt.grid(True)
plt.savefig("ROC.pdf")
plt.show()

feature_importance = dt_classifier.feature_importances_

features = X.columns

importance_df = pd.DataFrame(
    {'feature': features,
     'importance': feature_importance
     }).sort_values('importance', ascending=False)

print("特征重要性")
print(importance_df)
plt.figure(figsize=[20,12])
plt.barh(importance_df['feature'], importance_df['importance'])
plt.xlabel('重要性')
plt.title('特征重要性')
plt.gca().invert_yaxis()
plt.savefig("特征重要性可视化排序.pdf")
plt.show()