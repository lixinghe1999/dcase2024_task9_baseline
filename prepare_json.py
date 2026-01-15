"""{
    "data": [
     {
      "wav": "path_to_audio_file",
      "caption": "textual_desciptions"
     }
    ]
}"""
import os
import pandas as pd
import json
clothv2_path = "../dataset/clothv2/"
fsd50k_path = "../dataset/fsd50k/"
spatial_clothv2_path = "../dataset/simulation_clip/smartglass_clothv2_4/"

# clothv2
splits = ['development', 'evaluation', 'validation']

for split in splits:
    data = {"data": []}
    df = pd.read_csv(os.path.join(clothv2_path, f"clotho_captions_{split}.csv"))
    for index, row in df.iterrows():
        audio_file, captions = row['file_name'], row[['caption_1', 'caption_2', 'caption_3', 'caption_4', 'caption_5']]
        for caption in captions:
            data["data"].append({
                "wav": os.path.join(clothv2_path, split, audio_file),
                "caption": caption
            })
    with open(os.path.join('datafiles', f"clotho_{split}.json"), 'w') as f:
        json.dump(data, f, indent=4)

# fsd50k
splits = ['eval', 'dev']
for split in splits:
    json_path = os.path.join(fsd50k_path, f"fsd50k_{split}_auto_caption.json")
    data = json.load(open(json_path, 'r'))
    for i in range(len(data['data'])):
        data['data'][i]['wav'] = os.path.join(fsd50k_path, f'FSD50K.{split}_audio', data['data'][i]['wav'])
    with open(os.path.join('datafiles', f"fsd50k_{split}.json"), 'w') as f:
        json.dump(data, f, indent=4)

# spatial_clothv2
splits = ['train', 'test']
for split in splits:
    json_path = os.path.join(spatial_clothv2_path, f"{split}/metadata.json")
    data = json.load(open(json_path, 'r'))
    new_data = {"data": []}
    for i in range(len(data)):
        for j in range(len(data[i])):
            _data = data[i][j]
            new_data['data'].append({
                "wav": os.path.join(spatial_clothv2_path, split, 'audio', _data['audio_path']),
                "caption": _data['class']
            })
    with open(os.path.join('datafiles', f"spatial_clothv2_{split}.json"), 'w') as f:
        json.dump(new_data, f, indent=4)