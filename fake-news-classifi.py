{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "dcd926d1",
   "metadata": {},
   "outputs": [],
   "source": [
    "#libraries to import\n",
    "import pandas as pd\n",
    "from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer\n",
    "import re\n",
    "from nltk.corpus import stopwords\n",
    "from nltk.stem.porter import PorterStemmer\n",
    "from sklearn.feature_extraction.text import CountVectorizer\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.naive_bayes import MultinomialNB\n",
    "from sklearn import metrics\n",
    "import numpy as np\n",
    "import itertools"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "667f79e8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>title</th>\n",
       "      <th>author</th>\n",
       "      <th>text</th>\n",
       "      <th>label</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>House Dem Aide: We Didn’t Even See Comey’s Let...</td>\n",
       "      <td>Darrell Lucus</td>\n",
       "      <td>House Dem Aide: We Didn’t Even See Comey’s Let...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>FLYNN: Hillary Clinton, Big Woman on Campus - ...</td>\n",
       "      <td>Daniel J. Flynn</td>\n",
       "      <td>Ever get the feeling your life circles the rou...</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2</td>\n",
       "      <td>Why the Truth Might Get You Fired</td>\n",
       "      <td>Consortiumnews.com</td>\n",
       "      <td>Why the Truth Might Get You Fired October 29, ...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3</td>\n",
       "      <td>15 Civilians Killed In Single US Airstrike Hav...</td>\n",
       "      <td>Jessica Purkiss</td>\n",
       "      <td>Videos 15 Civilians Killed In Single US Airstr...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>4</td>\n",
       "      <td>Iranian woman jailed for fictional unpublished...</td>\n",
       "      <td>Howard Portnoy</td>\n",
       "      <td>Print \\nAn Iranian woman has been sentenced to...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   id                                              title              author  \\\n",
       "0   0  House Dem Aide: We Didn’t Even See Comey’s Let...       Darrell Lucus   \n",
       "1   1  FLYNN: Hillary Clinton, Big Woman on Campus - ...     Daniel J. Flynn   \n",
       "2   2                  Why the Truth Might Get You Fired  Consortiumnews.com   \n",
       "3   3  15 Civilians Killed In Single US Airstrike Hav...     Jessica Purkiss   \n",
       "4   4  Iranian woman jailed for fictional unpublished...      Howard Portnoy   \n",
       "\n",
       "                                                text  label  \n",
       "0  House Dem Aide: We Didn’t Even See Comey’s Let...      1  \n",
       "1  Ever get the feeling your life circles the rou...      0  \n",
       "2  Why the Truth Might Get You Fired October 29, ...      1  \n",
       "3  Videos 15 Civilians Killed In Single US Airstr...      1  \n",
       "4  Print \\nAn Iranian woman has been sentenced to...      1  "
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Data Read & Verificatoin\n",
    "df = pd.read_csv(\"D:\\\\NLP Data\\\\fake-news-classifi\\\\fake-news\\\\train.csv\")\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "5883f7e7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>title</th>\n",
       "      <th>author</th>\n",
       "      <th>text</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>House Dem Aide: We Didn’t Even See Comey’s Let...</td>\n",
       "      <td>Darrell Lucus</td>\n",
       "      <td>House Dem Aide: We Didn’t Even See Comey’s Let...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>FLYNN: Hillary Clinton, Big Woman on Campus - ...</td>\n",
       "      <td>Daniel J. Flynn</td>\n",
       "      <td>Ever get the feeling your life circles the rou...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2</td>\n",
       "      <td>Why the Truth Might Get You Fired</td>\n",
       "      <td>Consortiumnews.com</td>\n",
       "      <td>Why the Truth Might Get You Fired October 29, ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3</td>\n",
       "      <td>15 Civilians Killed In Single US Airstrike Hav...</td>\n",
       "      <td>Jessica Purkiss</td>\n",
       "      <td>Videos 15 Civilians Killed In Single US Airstr...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>4</td>\n",
       "      <td>Iranian woman jailed for fictional unpublished...</td>\n",
       "      <td>Howard Portnoy</td>\n",
       "      <td>Print \\nAn Iranian woman has been sentenced to...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   id                                              title              author  \\\n",
       "0   0  House Dem Aide: We Didn’t Even See Comey’s Let...       Darrell Lucus   \n",
       "1   1  FLYNN: Hillary Clinton, Big Woman on Campus - ...     Daniel J. Flynn   \n",
       "2   2                  Why the Truth Might Get You Fired  Consortiumnews.com   \n",
       "3   3  15 Civilians Killed In Single US Airstrike Hav...     Jessica Purkiss   \n",
       "4   4  Iranian woman jailed for fictional unpublished...      Howard Portnoy   \n",
       "\n",
       "                                                text  \n",
       "0  House Dem Aide: We Didn’t Even See Comey’s Let...  \n",
       "1  Ever get the feeling your life circles the rou...  \n",
       "2  Why the Truth Might Get You Fired October 29, ...  \n",
       "3  Videos 15 Civilians Killed In Single US Airstr...  \n",
       "4  Print \\nAn Iranian woman has been sentenced to...  "
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Drop the label coloumn for further use\n",
    "X = df.drop('label', axis=1)\n",
    "X.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "1befc321",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    1\n",
       "1    0\n",
       "2    1\n",
       "3    1\n",
       "4    1\n",
       "Name: label, dtype: int64"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Get Target data into outout variable - y = label\n",
    "y = df['label']\n",
    "y.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "7b9e6dd3",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "b55dbe53",
   "metadata": {},
   "outputs": [],
   "source": [
    "#drop the nan rows/data\n",
    "df = df.dropna()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "a417436d",
   "metadata": {},
   "outputs": [],
   "source": [
    "#get the new updated data in variable\n",
    "messages = df.copy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "a6f190cf",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Reset the index, indexes disturbed due to dropping of Nan rows from dataset.\n",
    "messages.reset_index(inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "0c0ef48c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'15 Civilians Killed In Single US Airstrike Have Been Identified'"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "messages['title'][3]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "f1c1c1a4",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Stemming & Removing unwanted data\n",
    "\n",
    "ps = PorterStemmer()\n",
    "corpus = []"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "6a597675",
   "metadata": {},
   "outputs": [],
   "source": [
    "for i in range(0, len(messages)):\n",
    "    review = re.sub('[^a-zA-Z]', ' ', messages['title'][i])\n",
    "    review = review.lower()\n",
    "    review = review.split()\n",
    "    \n",
    "#List Comperhencing PRocesss-\n",
    "    review = [ps.stem(word) for word in review if not word in stopwords.words('english') ] \n",
    "    review = ' '.join(review)\n",
    "    corpus.append(review)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "2c6e1f0f",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Create Bag of Words\n",
    "\n",
    "cv = CountVectorizer(max_features=5000, ngram_range=(1,3))\n",
    "X = cv.fit_transform(corpus).toarray()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "e99906c8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(18285, 5000)"
      ]
     },
     "execution_count": 46,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "1aaa5ec4",
   "metadata": {},
   "outputs": [],
   "source": [
    "y = messages['label']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "d61d8b05",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(18285,)"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "2e280f10",
   "metadata": {},
   "outputs": [],
   "source": [
    "#model evaluation by confusion matrix\n",
    "\n",
    "def plot_confusion_matrix(cm, classes,\n",
    "                         normalize=False,\n",
    "                         title='Confusion Matrix',\n",
    "                         cmap=plt.cm.Blues):\n",
    "    plt.imshow(cm, interpolation='nearest',cmap=cmap)\n",
    "    plt.title(title)\n",
    "    plt.colorbar()\n",
    "    tick_marks = np.arrange(len(classes))\n",
    "    plt.xticks(tick_marks,classes, rotation=45)\n",
    "    plt.yticks(tick_marks, classes)\n",
    "    \n",
    "    if normalize:\n",
    "        cm = cm.astype('float')/ cm.sum(axis=1)[:, np.newaxis]\n",
    "        print(\"Normalized Confusion Matrix\")\n",
    "    else:\n",
    "        print(\"Without Normalized CM\")\n",
    "        \n",
    "    thresh = cm.max()/2\n",
    "    \n",
    "    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):\n",
    "        plt.text(j, i, cm[i,j],\n",
    "                horizontalalignment = \"center\",\n",
    "                color = \"white\" if cm[i,j] > thresh else \"black\")\n",
    "        \n",
    "        plt.tight_layout()\n",
    "        plt.ylabel('True Label')\n",
    "        plt.xlabel('Predicted Label')\n",
    "        \n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "f901e7b7",
   "metadata": {},
   "outputs": [],
   "source": [
    "#split the dataset into TEst and Traim data\n",
    "\n",
    "X_train, y_train, x_test, y_test = train_test_split(X,y, test_size=0.33, random_state=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "47f9b460",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((12250, 5000), (6035, 5000), (12250,), (6035,))"
      ]
     },
     "execution_count": 61,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train.shape, y_train.shape, x_test.shape, y_test.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "61b51058",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Naive Bayes Algorithm Working\n",
    "\n",
    "classifier = MultinomialNB()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "16fc9535",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((6035, 5000), (6035,))"
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_train.shape,y_test.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "be52229b",
   "metadata": {},
   "outputs": [
    {
     "ename": "ValueError",
     "evalue": "y should be a 1d array, got an array of shape (6035, 5000) instead.",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "\u001b[1;32m~\\AppData\\Local\\Temp/ipykernel_3316/3343061556.py\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mclassifier\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfit\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mX_train\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0my_train\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      2\u001b[0m \u001b[0mpred\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mclassifier\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mpredict\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mX_test\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      3\u001b[0m \u001b[0mscore\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mmetrics\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0maccuracy_score\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0my_test\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mpred\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      4\u001b[0m \u001b[0mprint\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"accuracy:   %0.3f\"\u001b[0m \u001b[1;33m%\u001b[0m\u001b[0mscore\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      5\u001b[0m \u001b[0mcm\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mmetrics\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mconfusion_matrix\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0my_test\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mpred\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\sklearn\\naive_bayes.py\u001b[0m in \u001b[0;36mfit\u001b[1;34m(self, X, y, sample_weight)\u001b[0m\n\u001b[0;32m    613\u001b[0m         \u001b[0mself\u001b[0m \u001b[1;33m:\u001b[0m \u001b[0mobject\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    614\u001b[0m         \"\"\"\n\u001b[1;32m--> 615\u001b[1;33m         \u001b[0mX\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0my\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_check_X_y\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mX\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0my\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    616\u001b[0m         \u001b[0m_\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mn_features\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mX\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mshape\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    617\u001b[0m         \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mn_features_\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mn_features\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\sklearn\\naive_bayes.py\u001b[0m in \u001b[0;36m_check_X_y\u001b[1;34m(self, X, y)\u001b[0m\n\u001b[0;32m    478\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    479\u001b[0m     \u001b[1;32mdef\u001b[0m \u001b[0m_check_X_y\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mX\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0my\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 480\u001b[1;33m         \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_validate_data\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mX\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0my\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0maccept_sparse\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;34m'csr'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    481\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    482\u001b[0m     \u001b[1;32mdef\u001b[0m \u001b[0m_update_class_log_prior\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mclass_prior\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mNone\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\sklearn\\base.py\u001b[0m in \u001b[0;36m_validate_data\u001b[1;34m(self, X, y, reset, validate_separately, **check_params)\u001b[0m\n\u001b[0;32m    430\u001b[0m                 \u001b[0my\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mcheck_array\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0my\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mcheck_y_params\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    431\u001b[0m             \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 432\u001b[1;33m                 \u001b[0mX\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0my\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mcheck_X_y\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mX\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0my\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mcheck_params\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    433\u001b[0m             \u001b[0mout\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mX\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0my\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    434\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\sklearn\\utils\\validation.py\u001b[0m in \u001b[0;36minner_f\u001b[1;34m(*args, **kwargs)\u001b[0m\n\u001b[0;32m     70\u001b[0m                           FutureWarning)\n\u001b[0;32m     71\u001b[0m         \u001b[0mkwargs\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mupdate\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m{\u001b[0m\u001b[0mk\u001b[0m\u001b[1;33m:\u001b[0m \u001b[0marg\u001b[0m \u001b[1;32mfor\u001b[0m \u001b[0mk\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0marg\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mzip\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0msig\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mparameters\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0margs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m}\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 72\u001b[1;33m         \u001b[1;32mreturn\u001b[0m \u001b[0mf\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     73\u001b[0m     \u001b[1;32mreturn\u001b[0m \u001b[0minner_f\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     74\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\sklearn\\utils\\validation.py\u001b[0m in \u001b[0;36mcheck_X_y\u001b[1;34m(X, y, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, estimator)\u001b[0m\n\u001b[0;32m    805\u001b[0m                         ensure_2d=False, dtype=None)\n\u001b[0;32m    806\u001b[0m     \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 807\u001b[1;33m         \u001b[0my\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mcolumn_or_1d\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0my\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mwarn\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mTrue\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    808\u001b[0m         \u001b[0m_assert_all_finite\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0my\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    809\u001b[0m     \u001b[1;32mif\u001b[0m \u001b[0my_numeric\u001b[0m \u001b[1;32mand\u001b[0m \u001b[0my\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mdtype\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mkind\u001b[0m \u001b[1;33m==\u001b[0m \u001b[1;34m'O'\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\sklearn\\utils\\validation.py\u001b[0m in \u001b[0;36minner_f\u001b[1;34m(*args, **kwargs)\u001b[0m\n\u001b[0;32m     70\u001b[0m                           FutureWarning)\n\u001b[0;32m     71\u001b[0m         \u001b[0mkwargs\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mupdate\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m{\u001b[0m\u001b[0mk\u001b[0m\u001b[1;33m:\u001b[0m \u001b[0marg\u001b[0m \u001b[1;32mfor\u001b[0m \u001b[0mk\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0marg\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mzip\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0msig\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mparameters\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0margs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m}\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 72\u001b[1;33m         \u001b[1;32mreturn\u001b[0m \u001b[0mf\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     73\u001b[0m     \u001b[1;32mreturn\u001b[0m \u001b[0minner_f\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     74\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\sklearn\\utils\\validation.py\u001b[0m in \u001b[0;36mcolumn_or_1d\u001b[1;34m(y, warn)\u001b[0m\n\u001b[0;32m    843\u001b[0m         \u001b[1;32mreturn\u001b[0m \u001b[0mnp\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mravel\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0my\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    844\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 845\u001b[1;33m     raise ValueError(\n\u001b[0m\u001b[0;32m    846\u001b[0m         \u001b[1;34m\"y should be a 1d array, \"\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    847\u001b[0m         \"got an array of shape {} instead.\".format(shape))\n",
      "\u001b[1;31mValueError\u001b[0m: y should be a 1d array, got an array of shape (6035, 5000) instead."
     ]
    }
   ],
   "source": [
    "classifier.fit(X_train,y_train)\n",
    "pred = classifier.predict(X_test)\n",
    "score = metrics.accuracy_score(y_test, pred)\n",
    "print(\"accuracy:   %0.3f\" %score)\n",
    "cm = metrics.confusion_matrix(y_test, pred)\n",
    "plot_confusion_matrix(cm,classes=['FAKE', 'REAL'])\n",
    "#Note: Test Train Split so Model is not performing with no error"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8a2d14c6",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
