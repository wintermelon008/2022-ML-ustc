# Machine Learning Lab2

SVM

By Yanwu Gu 2022.10.12

## 1. Theory of Supported Vector Machine

You can refer to the ppt or the Chap 6 of the textbook.

## 2. Data

In order to simplify the lab, we give the function `generate_data(dim, num)` for you to freely generate the data. The data was linearly separable, but added some mistakes intentionally. Features, labels and rate of mislabel will be given by the function respectively.

You do not need to modify the function`generate_data(dim, num)`.

## 3. Tasks, Tips and Requirements

### 3.1 Tasks

You are required to complete the class `SVM1` and `SVM2` using different methods to find the solution of the supported vector machine. More specificly, since the key of solving SVM is to solve the quadratic programming problem (6.6) in your textbook, you just need to use **two** methods to solve (6.6). The remaining part like predict can be the same. 

After finishing the SVM class, you need to test the efficiency of your code. The comparison must include

1. The accuracy,
2. The time of culculation (trainning),

If possible, you can use `sklearn` to compare with your code, feel free to be beaten by it. 

### 3.2 Tips

There are some tips for the lab:

1. We do not recommend you to use existing function to solve the **quadratic programming** problem directly, which will be penalised. Of course, if you cannot complete two methods from scratch, you can use library function.
2. We recommend you to use proper dims to make sure your result reliable, and different dims or numbers of examples will let your report rich in content. But do not let it verbose.
3. Since our data is based on linear kernel, you do not need to try other kernels. But you can try soft margin or regularization to improve the ability of your model. Remember it's not the key point of this lab. 
4. Remember to add your **mislabel rate**, which is generate by the function `generate_data` for us.

###  3.3 Requirements

- **Do not** use sklearn or other machine learning library, you are only permitted with numpy, pandas, matplotlib, and [Standard Library](https://gitee.com/link?target=https%3A%2F%2Fdocs.python.org%2F3%2Flibrary%2Findex.html), you are required to **write this project from scratch.**
- You are allowed to discuss with other students, but you are **not allowed to plagiarize the code**, we will use automatic system to determine the similarity of your programs, once detected, both of you will get **zero** mark for this project.

## 4. Submission

* Report

  * The method you use, and briefly talk about its principle
  * The result of your methods
  * The comparison of your methods

* Submit a .zip file with following contents

  --main.ipynb

  --Report.pdf

* Please name your file as `LAB2_PBXXXXXXXX_NAME.zip`, **for wrongly named file, we will not count the mark**

* Sent an email to [ml_2022_fall@163.com](mailto:ml_2022_fall@163.com) with your zip file before deadline

* **Deadline: 2022.10.30 23:59:59** 

* For late submission, please refer to [this](https://gitee.com/Sqrti/ml_2022_f#一关于课程)