# Data information

## Discussions_data

This folder shall be contained in `data` folder.

### Data Description

This folder have json files. Json structure is like below:

- *Tags*: Array, contains tags on the post as type, name, unique tag_id


```json
[
{
"type":"specialty",
"name":"Anesthesia",
"tag_id":64,
"is_taggable":false
},
{
"type":"specialty",
"name":"Cardiology",
"tag_id":70,
"is_taggable":false
},..]
```

- *Full Description*: Text, description of post/actual post


```json
"full_description":"Ex smoker, having breathlessness on exertion, having following reports. ABG is in room air. Suggest treatment."
```

- *No. Helpful*: Integer, How many people/doctors tagged the post as helpful


```json
"no_helpful":2
```

- *No. Answers*: Integer, Number of answers on the post


```json
"no_answers":17
```

- *User* : Json, Contains information related to user who is author of post


```json
"user":{
"username":"tsarkar1",
"specialty":{
"is_main_specialty":true,
"name":"Pulmonology",
"specialty_name":"Pulmonology",
"specialty_id":79,
"type":"specialty",
"id":79
},
"profile_pic_url":"https://media.curofy.com/7a14f500da0d4201a9ba35f8c344c020_1.jpg",
"country_code":"+91",
"no_followers":214,
"display_name":"Dr. Tapan Sarkar",
"no_cases":54,
"leaderboard_score":559,
"no_following":35,
"no_answers":587,
"predicted_gender":1
}
```

- *Images* : Json, Contains information and images itself related to post/ posted by author


```json
"images":[
{
"small_width":612,
"image_color":"#eceff1",
"lossless_width":612,
"small_height":558,
"height":558,
"image_id":11243,
"width":612,
"url":"https://media.curofy.com/13903.736f561a08a652055553b43b56b1c76a.jpg",
"lossless_height":558,
"small_url":"https://media.curofy.com/13903.736f561a08a652055553b43b56b1c76a.jpg",
"lossless_url":"https://media.curofy.com/13903.736f561a08a652055553b43b56b1c76a.jpg",
"image_caption":"",
"format_type":"jpeg"
},..]
```

- *Type* : Text, Type of post


```json
"type":"Advertisement"
```

## Data Structuring & Cleaning Steps

Following script `data/dataCleaningModel.py`

- Read all json files from the folder using pandas

```python
for i in listdir(self.dir_path):
    if i.endswith(".json"):
        data = pd.read_json(self.dir_path+i)
```

- Looping through the json for structuring data row by row

```python
for i,j,k,l in zip(data.tags,data.full_description,data.type,data.user):
```

- Normalizing `tags` json and creating a dataframe

```python
a = json_normalize(i)
a = a[a.type=="specialty"][["name","tag_id"]]
```

- Creating a final data frame. Keeping only Tags, Case Description, Tags and their unique IDs

```python
final_data = pd.DataFrame({"case_description":description, "tag":tags, "tag_id":tags_id,
                           "user_specialty":user_specialty, "type":types})
final_data = pd.concat([final_data,pd.get_dummies(final_data.tag_id.apply(pd.Series).stack()).sum(level=0)],
                       axis=1)
```

- Cleaning text of data like removing URLs, removing extra characters etc

```python
def remove_hash(self,x):
    return x.replace("#", "")

def remove_urls(self,x):
    return re.sub(r"http\S+", "", x)

def clean_char(self,x):
    return re.sub('[\W_]+', ' ', x)

def data_cleaning(self,data):
    data.case_description = data.case_description.apply(self.remove_hash)
    data.case_description = data.case_description.apply(self.remove_urls)
    data.case_description = data.case_description.apply(self.clean_char)
    data.case_description = data.case_description.str.lower()
    return data
```


