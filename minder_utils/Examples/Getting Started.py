'''
<h1>Getting Started</h1>
To use the package, please:

1. access RDS to download the necessary files including mapping.json, validated_date.py
    and random_id_to_research_id.json (optional, for TIHM data).

2. acess the research portal to get the access token

3. Optional, download the TIHM data from research portal.

4. use the ```Get Started.ipynb``` to set the paths to the files and set the token.

5. Check the examples in ```Instruction.ipynb```.

<h1>Introduction</h1>
Currently, the package provides the functions:
<ul>
  <li>download: download and refresh the data</li>
  <li>download weekly data: download and refresh the data weekly</li>
  <li>loading the raw data: the data (both tihm and dri) will be re-format to a standard format</li>
  <li>loading the weekly data: the data (both tihm and dri) will be processed in weekly format
  and provided engineering features. </li>
  <li>pre-processing data: the data (both tihm and dri) will be processed (aggregated hourly/daily) into array,
  which is ready for machine learning models (note normalisation techniques need to be specified).
  </li>
</ul>

Here is an overview of this package,

1. ```Downloader```
    <p><b>Intro</b>:
        ```Downloader``` can download all types of data and
        refresh the downloaded data.<p>
    <p><b>Usage</b>:

    ```
    from download.download import Downloader
    Downloader().export(since='2021-10-10', until='2021-10-12',
    reload=True, save_path='./data/activity/', categories=['raw_activity_pir'])
    ```


2. ```Formatting```

    <p><b>Intro</b>:

    ```Formatting``` will reformat the data to a dataframe contains
     ```['id', 'time', 'location', 'value']```. Furthermore, it can automatically
     change the TIHM data same as DRI and concatenate them (optional).</p>

     <p><b>Usage</b>:

     ```
     from minder_utils.formatting import Formatting
    formater = Formatting()
    print(formater.activity_data)
     ```</p>

3. ```Feature_engineer```

    <p><b>Intro</b>:

    ```Feature_engineer``` will reformat the data to a dataframe contains
     ```['id', 'time', 'location', 'value', 'week']```, where week is the week index
        of a date, e.g. the week for 2020.01.01 is 2001 (first week of the year 20).
        Furthermore, it can statistically analysis the weekly data. Please check the
        documentation for details.</p>

     <p><b>Usage</b>:

     ```
     from minder_utils.feature_engineering import Feature_engineer
    fe = Feature_engineer(Formatting())
    print(fe.activity)
     ```</p>

<h1>Configurations</h1>

In the ```./configurations```, there are two editable files:

<ul>
  <li>```config_dri.yaml```: configurations for the data</li>
  <li>```config_engineering_feature.yaml```: configurations for the feature engineering</li>
</ul>

Here's some important attributes you man changed according to your need. In the table below,
dri is the ```config_dri.yaml```, fe is the ```config_engineering_feature.yaml```.


<table>
    <colgroup>
       <col span="1" style="width: 15%;">
       <col span="1" style="width: 15%;">
       <col span="1" style="width: 70%;">
    </colgroup>
<tbody>
  <tr>
    <th>config name</th>
    <th>which file</th>
    <th>description</th>
  </tr>
  <tr>
    <td>save_path</td>
    <td>both</td>
    <td>the path to save the data so you won't need to process it next time.</td>
  </tr>
  <tr>
    <td>save_name</td>
    <td>both</td>
    <td>the name to save the data. <b>NOTE</b>: RECOMMEND NOT TO CHANGE. If you want to
    save the data in a different name, please edit the ```save``` attribute under
    corresponding item.</td>
  </tr>
  <tr>
    <td>verbose</td>
    <td>both</td>
    <td>print the message in console</td>
  </tr>
  <tr>
    <td>refresh</td>
    <td>both</td>
    <td>use the processed data or re-process the data again. <b>NOTE</b>: if you use
    ```weekly_loader``` to refresh the data every week, please set it as True</td>
  </tr>
  <tr>
    <td>add_tihm</td>
    <td>dri</td>
    <td>concatenate the TIHM data to the DRI data.</td>
  </tr>
  <tr>
    <td>nocturia</td>
    <td>fe</td>
    <td>The time range to calculate the activities of bathroom during the night.
    The time outside this range will be used to calculate the bathroom activity
    during the day.</td>
  </tr>
  <tr>
    <td>activity</td>
    <td>fe</td>
    <td>What attributes will be returned to calculate the weekly activities (used for
     training the models)</td>
  </tr>
</tbody>
</table>

The other attributes basically can explain themselves.

<h1>Queries</h1>
If you have any problems while using it, please reach us on slack or create an issue on git. Thanks!
'''

