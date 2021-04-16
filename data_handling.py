import pandas as pd
import sklearn
from pandas.io.json import json_normalize
from numpy import loadtxt
from sklearn import preprocessing
from turfpy import measurement
from geojson import Point, Polygon, Feature
import requests


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

    mtd_df_output["query_time"] = pd.to_datetime(mtd_df_output["query_time"].astype(str).str[:-6])
    
    mtd_df_output["scheduled_date_only"] = mtd_df_output["scheduled"].astype(str).str[:10]
    
    
    mtd_df_output["expected_scheduled_diff"] = mtd_df_output['expected'].subtract(mtd_df_output['scheduled']).astype('timedelta64[s]')
    mtd_df_output["diff_in_sec"] = mtd_df_output['actual'].subtract(mtd_df_output['expected']).astype('timedelta64[s]')
    
    print("DATA CLEANING complete")

    return mtd_df_output


# This function has been succeeded by rolling_correlations(). Should we delete it?
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

    suffix = "_avg_by_" + group

    clean_dataframe["query_time_parsed"] = pd.to_datetime(clean_dataframe["query_time"])
    
    clean_dataframe = clean_dataframe.merge(
        
        # TODO: Separate this one-liner into several steps and pass the results back to the merge() function above
        clean_dataframe.groupby(group).resample('2H', on='query_time_parsed').mean()['diff_in_sec'].shift()
        .rolling(12, min_periods=1).mean().fillna(-10000).reset_index(),

        on = [group, 'query_time_parsed'], how = "left", suffixes = ("", suffix)
        
    )

    clean_dataframe.drop(['query_time_parsed'], axis=1)

    return clean_dataframe
    #TODO: groupby(), then use iloc() to select the previous rows, think back to previous backfilling functions
    # Go thru each row, take iloc() to slice the rows in for instance by bits of the last 10 rows 
    # for each_idx, each_df in clean_dataframe.groupby(group):
    # try groupby(["stop_id_stem", "headsign", "trip_shape_id", "vehicle_id", "scheduled"]
    # AVOID masks like df[df["col_x"] < 100] cos it'll slow down things

    #     expected_timestamp = pd.to_datetime(each_df["expected"].astype(str))
    #     actual_timestamp = pd.to_datetime(each_df["actual"].astype(str))

    #     lateness_series = actual_timestamp.subtract(expected_timestamp).astype('timedelta64[s]')
    #     avg_lateness = lateness_series.rolling(100, min_periods=1).mean()
    #     clean_dataframe.loc[each_df.index, new_feature_name] = avg_lateness


def obtain_weekday(clean_dataframe):
    print("Adding Weekday (day_of_week and query_weekday_encoded)")
    le = preprocessing.LabelEncoder()
    clean_dataframe['query_time'] = pd.to_datetime(clean_dataframe['query_time'])
    clean_dataframe['day_of_week'] = clean_dataframe["query_time"].dt.day_name()
    clean_dataframe['query_weekday_encoded'] = le.fit_transform(clean_dataframe.day_of_week.values)

    clean_dataframe = clean_dataframe.loc[:,~clean_dataframe.columns.str.startswith('DOW')]
    clean_dataframe = pd.concat([clean_dataframe, pd.get_dummies(clean_dataframe.day_of_week, prefix='DOW')], 1)


    print("Adding Weekday (day_of_week and query_weekday_encoded) complete")

    return clean_dataframe

def part_of_day_lambda(x):
    if (x > 4) and (x <= 8):
        return 'Early Morning'
    elif (x > 8) and (x <= 12 ):
        return 'Morning'
    elif (x > 12) and (x <= 16):
        return'Noon'
    elif (x > 16) and (x <= 20) :
        return 'Eve'
    elif (x > 20) and (x <= 24):
        return'Night'
    elif (x <= 4):
        return'Late Night'
    # TODO: Do one-hot encoding. It is important to have enough signal to divide parts of days
    # Do the same for Day Of Week, convert it from its current 0~6 system to one-hot encoding
    # For tree-based models, this makes big difference

def obtain_part_of_day(clean_dataframe):
    hours_series = pd.to_datetime(clean_dataframe["query_time"]).dt.hour

    clean_dataframe['part_of_day'] = hours_series.apply(part_of_day_lambda)
    clean_dataframe = clean_dataframe.loc[:,~clean_dataframe.columns.str.startswith('POD')]

    clean_dataframe = pd.concat([clean_dataframe, pd.get_dummies(clean_dataframe.part_of_day, prefix='POD')], 1)

    return clean_dataframe


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


def obtain_time_elapsed_since_twelve_AM(clean_dataframe):
    print("Adding Time of Day In Seconds (time_elapsed_since_twelve_AM)")
    clean_dataframe['query_time'] = pd.to_datetime(clean_dataframe['query_time'])
    clean_dataframe["time_elapsed_since_twelve_AM"] = (clean_dataframe["query_time"].dt.hour * 60 + clean_dataframe["query_time"].dt.minute) * 60 + clean_dataframe["query_time"].dt.second

    print("Adding Time of Day In Seconds (time_elapsed_since_twelve_AM) complete")

    return clean_dataframe


def backfill_direct_distance_to_stop(clean_dataframe, lookup):

    def distance_calculation(row):
        start = Feature(geometry=Point(lookup[row["stop_id"]]))
        end = Feature(geometry=Point((row["location_long"], row["location_lat"])))
        return measurement.distance(start, end)
  
    clean_dataframe["distance_to_stop"] = clean_dataframe.apply(distance_calculation, axis=1)

    return clean_dataframe


def backfill_polygon_intersection(clean_dataframe, polygon, polygon_num):

    def inclusion_check(row):
        #print(row)
        #print(type(row))
        location = Feature(geometry=Point((row["location_long"], row["location_lat"])))
        return measurement.boolean_point_in_polygon(location, polygon)
  
    column_name = "in_polygon_" + str(polygon_num)

    clean_dataframe[column_name] = clean_dataframe.apply(inclusion_check, axis=1)

    return clean_dataframe


def geographic_features(clean_dataframe):
    print("Backfilling of geo columns STARTING")
    green_st_polygon = Feature(geometry=Polygon([[(-88.219423, 40.112805), (-88.219294, 40.109104), 
    (-88.238732, 40.109104), (-88.238732, 40.112652)]]))
    champaign_polygon = Feature(geometry=Polygon([[(-88.238538, 40.119902), (-88.238538, 40.114400), 
    (-88.245212, 40.114400), (-88.245212, 40.119902)]]))
    
    clean_dataframe = backfill_polygon_intersection(clean_dataframe, green_st_polygon, 1)
    clean_dataframe = backfill_polygon_intersection(clean_dataframe, champaign_polygon, 2)
    request_str = 'https://developer.cumtd.com/api/v2.2/json/getstops'
    key = "8ad71215eee445679382e558275ec266"
    params = {'key': key}
    fetched_stops = requests.get(request_str, params=params)

    stops_dict = {}
    for each_stop in fetched_stops.json()["stops"]:
        for each_stop_point in each_stop["stop_points"]:
            # print(each_stop_point["code"],each_stop_point["stop_id"],each_stop_point["stop_lat"],each_stop_point["stop_lon"])
            stops_dict[str(each_stop_point["stop_id"])] = (each_stop_point["stop_lon"], each_stop_point["stop_lat"])
    # What should lookup be?
    # Keys = id for every stop
    # Values = long, lat coordinates for every stop

    # how to get keys and values?
    # Query MTD API GetStops method
    # Imagine return from GetStops as a dictionary (stops_dict)
    ## access stop_id and lat + long coordinates for every stop point in stop["stop_points"] for every stop in stop_dict["stops"]

    clean_dataframe = backfill_direct_distance_to_stop(clean_dataframe, stops_dict)
    print("Backfilling of geo columns COMPLETE")

    return clean_dataframe


def data_handling_main(filename_string, suffix_str):
    cumtd_df = pd.read_csv(filename_string + '.csv')
    cumtd_df = backfill_dataframe_actual_time(cumtd_df, filename_string)
    cumtd_df = data_cleaning(cumtd_df)
    cumtd_df = backfill_time_since_appearance(cumtd_df, filename_string)
    cumtd_df = obtain_number_of_stalls(cumtd_df)
    cumtd_df = obtain_weekday(cumtd_df)
    cumtd_df = obtain_part_of_day(cumtd_df)
    cumtd_df = obtain_time_elapsed_since_twelve_AM(cumtd_df)
    cumtd_df = rolling_correlations(cumtd_df, "expected_mins", "time_since_appearance")
    cumtd_df = per_bus_rolling_average(cumtd_df, "number_of_stalls")
    cumtd_df = per_bus_rolling_average(cumtd_df, "expected_scheduled_diff")
    cumtd_df =  geographic_features(cumtd_df)

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