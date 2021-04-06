import pandas as pd
import sklearn
from pandas.io.json import json_normalize
from numpy import loadtxt
from sklearn import preprocessing


def import_test():
    print("IMPORT IS WORKING")

def backfill_dataframe_actual_time(unfilled_dataframe, file_name_string):
    print("Backfilling Actual Time ('actual') BEGINS.")
    
    unfilled_dataframe["stop_id_stem"] = unfilled_dataframe['stop_id'].str.split(':').str[0]
    for each_idx, each_df in unfilled_dataframe.groupby(["stop_id_stem", "headsign", 
                                                                "trip_shape_id", "vehicle_id", 
                                                                "scheduled"]):
        # print(each_idx, each_df)
    
        unfilled_dataframe.loc[each_df.index,  "actual"] = each_df["query_time"].iloc[-1]
        # print("Backfilling with~", each_df["query_time"].iloc[-1])
    
    print("Backfilling Actual Time ('actual') complete.")
    return unfilled_dataframe

def backfill_time_since_appearance(inspected_df, file_name_string):
    print("Backfilling TSA BEGINS.")
    for each_idx, each_df in inspected_df.groupby(["stop_id_stem", "headsign", "trip_shape_id", "vehicle_id", "scheduled"]):
        first_row_time = each_df.iloc[0]["query_time"]
        time_since_appearance = (each_df.query_time - first_row_time).astype('timedelta64[s]')
        inspected_df.loc[each_df.index, 'time_since_appearance'] = time_since_appearance
        # print("Fold backfilled:", each_idx)

        # Work In Progress
        # Create a new value which is an in-row correlation "time_since_appearance" & "expected_mins"
        # NoteToSelf: Might be a for-loop but do investigate if Pandas can do it
    print("Backfilling TSA complete.")
    return inspected_df


def data_cleaning(mtd_df):
    # mtd_df = mtd_df.drop(["is_departed"], axis = 1).dropna()
    print("DATA CLEANING BEGINS")
    mtd_df_output = mtd_df[mtd_df.vehicle_id != 'null']
    le = preprocessing.LabelEncoder()
    mtd_df_output['stop_id_encoded'] = le.fit_transform(mtd_df.stop_id.values)
    mtd_df_output['headsign_encoded'] = le.fit_transform(mtd_df.headsign.values)
    mtd_df_output['trip_shape_id_encoded'] = le.fit_transform(mtd_df.trip_shape_id.values)
    mtd_df_output["scheduled"] = pd.to_datetime(mtd_df_output["scheduled"].astype(str).str[:-6])
    mtd_df_output["expected"] = pd.to_datetime(mtd_df_output["expected"].astype(str).str[:-6])
    mtd_df_output["actual"] = pd.to_datetime(mtd_df_output["actual"].astype(str).str[:-5])

    # TODO: {{}}

    mtd_df_output["query_time"] = pd.to_datetime(mtd_df_output["query_time"].astype(str).str[:-6])
    
    mtd_df_output["scheduled_date_only"] = mtd_df_output["scheduled"].astype(str).str[:10]
    
    
    mtd_df_output["expected_scheduled_diff"] = mtd_df_output['expected'].subtract(mtd_df_output['scheduled']).astype('timedelta64[s]')
    mtd_df_output["diff_in_sec"] = mtd_df_output['actual'].subtract(mtd_df_output['expected']).astype('timedelta64[s]')
    
    print("DATA CLEANING complete")

    return mtd_df_output


# FUNCTION WORK-IN-PROGRESS
def obtain_correlation(clean_subframes, filename_string):
    clean_subframes["corr_expected_min_vs_tsa"] = 0
    for each_idx, each_df in clean_subframes.groupby(["stop_id_stem", "headsign", "trip_shape_id", "vehicle_id", "scheduled"]):
        group_of_rows = []
        in_group_accumulator = 0
        for index, row in each_df.iterrows():
            in_group_accumulator += 1
            group_of_rows.append(row)
            group_df = pd.DataFrame(group_of_rows)
            if in_group_accumulator > 2:
                #clean_subframes.loc[clean_subframes['index'] == row["index"]]['corr_expected_min_vs_tsa'] = group_df['expected_mins'].corr(group_df['time_since_appearance'])
                # print(group_df['expected_mins']) # DEBUG
                clean_subframes.loc[index, 'corr_expected_min_vs_tsa'] = (group_df['expected_mins'] * 60).corr(group_df['time_since_appearance'])
            #if pd.isnull(clean_subframes.loc[index, 'corr_expected_min_vs_tsa']):
                # print(group_df[['expected_mins', 'time_since_appearance']])
    
    return clean_subframes

def rolling_correlations(clean_dataframe, feature_1, feature_2):

    new_feature_name = "corr_" + feature_1 + "_vs_" + feature_2
    clean_dataframe[new_feature_name] = -2

    for each_idx, each_df in clean_dataframe.groupby(["stop_id_stem", "headsign", "trip_shape_id", "vehicle_id", "scheduled"]):
        correlations = each_df[feature_1].rolling(each_df.shape[0], min_periods=1).corr(each_df[feature_2])
        clean_dataframe.loc[each_df.index, new_feature_name] = correlations
    
    clean_dataframe[new_feature_name] = clean_dataframe[new_feature_name].fillna(-2)

    return clean_dataframe

def obtain_number_of_stalls(clean_dataframe):
    
    print("Adding Number of Stalls (number_of_stalls)")
    clean_dataframe["number_of_stalls"] = pd.Series(0, index=clean_dataframe.index)
    
    for each_idx, each_df in clean_dataframe.groupby(["stop_id_stem", "headsign", "trip_shape_id", "vehicle_id", "scheduled"]):
        in_group_accumulator = 0
        current_expected_mins = -1
        first_row = True
        for index, row in each_df.iterrows():
            if row["expected_mins"] != current_expected_mins or first_row is True:
                
                in_group_accumulator = 0
                current_expected_mins = row["expected_mins"]

                if first_row is True:
                    first_row = False

            in_group_accumulator += 1

            clean_dataframe.loc[index, "number_of_stalls"] = in_group_accumulator

    print("Adding Number of Stalls (number_of_stalls) complete")
    return clean_dataframe

def obtain_avg_lateness_by_group(clean_dataframe, group: str):

    new_feature_name = "avg_lateness_by_" + group
    clean_dataframe[new_feature_name] = -10000

    for each_idx, each_df in clean_dataframe.groupby(group):

        expected_timestamp = pd.to_datetime(each_df["expected"].astype(str))
        actual_timestamp = pd.to_datetime(each_df["actual"].astype(str))

        lateness_series = actual_timestamp.subtract(expected_timestamp).astype('timedelta64[s]')
        avg_lateness = lateness_series.rolling(100, min_periods=1).mean()
        clean_dataframe.loc[each_df.index, new_feature_name] = avg_lateness

    # Think about the filtering NOW

    return clean_dataframe
 
# TODO: 1. def obtain_avg_lateness()
# TODO: 2. Obtain the rolling average of ["expected_scheduled_diff"] and ["number_of_stalls"] per ["vehicle_id"]

def per_bus_rolling_average(clean_dataframe, feature):

    new_feature_name = feature + "_rolling_avg"
    clean_dataframe[new_feature_name] = pd.Series(-1, index=clean_dataframe.index)

    for each_idx, each_df in clean_dataframe.groupby("vehicle_id"):
        #print(each_df[feature])
        TTT = each_df[feature].rolling(100, min_periods=1).mean()
        clean_dataframe.loc[each_df.index, new_feature_name] = TTT
        # print(TTT)    
        # clean_dataframe.iloc[each_df.index][new_feature_name] = TTT
    
    return clean_dataframe


    # group by vehicle number
    # calculate the rolling avg for feature within each group subframe
    # assign rolling avg to clean_dataframe

def obtain_weekday(clean_dataframe):
    print("Adding Weekday (day_of_week and query_weekday_encoded)")
    le = preprocessing.LabelEncoder()
    clean_dataframe['query_time'] = pd.to_datetime(clean_dataframe['query_time'])
    clean_dataframe['day_of_week'] = clean_dataframe["query_time"].dt.day_name()
    clean_dataframe['query_weekday_encoded'] = le.fit_transform(clean_dataframe.day_of_week.values)

    print("Adding Weekday (day_of_week and query_weekday_encoded) complete")

    return clean_dataframe


def obtain_time_elapsed_since_twelve_AM(clean_dataframe):
    print("Adding Time of Day In Seconds (time_elapsed_since_twelve_AM)")
    clean_dataframe['query_time'] = pd.to_datetime(clean_dataframe['query_time'])
    clean_dataframe["time_elapsed_since_twelve_AM"] = (clean_dataframe["query_time"].dt.hour * 60 + clean_dataframe["query_time"].dt.minute) * 60 + clean_dataframe["query_time"].dt.second

    print("Adding Time of Day In Seconds (time_elapsed_since_twelve_AM) complete")

    return clean_dataframe



def data_handling_main(filename_string, suffix_str):
    cumtd_df = pd.read_csv(filename_string + '.csv')
    cumtd_df = backfill_dataframe_actual_time(cumtd_df, filename_string)
    cumtd_df = data_cleaning(cumtd_df)
    cumtd_df = backfill_time_since_appearance(cumtd_df, filename_string)
    cumtd_df = obtain_number_of_stalls(cumtd_df)
    cumtd_df = obtain_weekday(cumtd_df)
    cumtd_df = obtain_time_elapsed_since_twelve_AM(cumtd_df)

    cumtd_df.to_csv(filename_string + suffix_str +".csv")

    return cumtd_df


if __name__ == '__main__':
    # main functionality
    filename_string = "cumtd_3_3_21"
    cumtd_df = pd.read_csv(filename_string + '.csv')
    cumtd_df = backfill_dataframe_actual_time(cumtd_df, filename_string)
    cumtd_df = data_cleaning(cumtd_df)
    cumtd_df = backfill_time_since_appearance(cumtd_df, filename_string)
    cumtd_df.to_csv(filename_string +"CLEANED_mar30"+".csv")