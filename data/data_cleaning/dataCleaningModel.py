import pandas as pd
from os import listdir
import json
import re
from pandas.io.json import json_normalize
from sklearn.preprocessing import OneHotEncoder
from text2digits import text2digits
import nltk
import pickle
nltk.download('wordnet')

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()


class dataCleaning(object):
    data_path = "data/discussions_data/text-annotated-data-curofy.csv"
    dir_path = "data/discussion_data_new/"
    t2d = text2digits.Text2Digits()

    def read_data(self):
        description = []
        tags = []
        tags_id = []
        types = []
        user_specialty = []
        for i in listdir(self.dir_path):
            if i.endswith(".json"):
                data = pd.read_json(self.dir_path + i)
                for i, j, k, l in zip(data.tags, data.full_description, data.type, data.user):
                    a = json_normalize(i)
                    a = a[a.type == "specialty"][["name", "tag_id"]]
                    description.append(j)
                    types.append(k)
                    tags.append(list(a.name))
                    tags_id.append(list(a.tag_id))
                    user_specialty.append(l["specialty"]["specialty_id"])
        final_data = pd.DataFrame({"case_description": description, "tag": tags, "tag_id": tags_id,
                                   "user_specialty": user_specialty, "type": types})
        cat_columns = ['tag_id']
        new_series = final_data.tag_id.apply(pd.Series).stack()
        frame = { 'tag_id': new_series}
        result = pd.DataFrame(frame)
        cat_df_processed = pd.get_dummies(result, prefix_sep="__",columns=cat_columns)
        df = cat_df_processed.sum(level=0)
        df_processed = pd.concat([final_data,df],axis=1)
        df_processed.columns= df_processed.columns.astype(str)

        #saving tag categories encoding for unseen test case usage
        cat_tag_dummies = list(df.columns)
        filename = 'tag_cat'
        outfile = open(filename,'wb')
        pickle.dump(cat_tag_dummies,outfile)
        outfile.close()
        # final_data = pd.concat([final_data, pd.get_dummies(final_data.tag_id.apply(pd.Series).stack()).sum(level=0)],
        #                        axis=1)
        # final_data.to_csv('try_final.csv')
        df_processed = df_processed.drop("tag", axis=1)
        df_processed = df_processed.drop("tag_id", axis=1)
        df_processed = df_processed.drop_duplicates().reset_index(drop=True)
        return self.data_cleaning(df_processed)

    def remove_hash(self, x):
        return x.replace("#", "")

    def remove_urls(self, x):
        return re.sub(r"http\S+", "", x)

    def clean_char(self, x):
        return re.sub('[\W_]+', ' ', x)

    def number_to_words(self, x):
        try:
            x = self.t2d.convert(x)
        except Exception as error:
            print(error)
        return x

    def data_cleaning(self, data):
        data.case_description = data.case_description.apply(self.remove_hash)
        data.case_description = data.case_description.apply(self.remove_urls)
        data.case_description = data.case_description.apply(self.clean_char)
        data.case_description = data.case_description.apply(self.number_to_words)
        data.case_description = data.case_description.str.lower()
        data.case_description = data.case_description.apply(self.lemmatize_text)
        print(data.columns)
        # data = data[["case_description","type"]]
        data = data.drop_duplicates().reset_index(drop=True)
        return data

    def lemmatize_text(self, text):
        return " ".join([lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)])
