#读取data文件夹中的所有气象数据和对应点位，进行模拟
import gc
import os
import sys
os.chdir(sys.path[0])
import pandas as pd
import geopandas as gpd
import concurrent.futures
from tqdm import tqdm
from datetime import datetime
from utils.modelingcrop import modelingcrop
from aquacrop import InitialWaterContent

soil_type_dic = {
    '3':'Clay',
    '5':'ClayLoam',
    '7':'SiltLoam',
    '9':'Loam',
    '10':'SandyClayLoam',
    '11':'SandyLoam',
    '12':'LoamySand'
}

def run_point_simulation(i, point_id, attr_list, point_coords, weather_dir):
    initWC = InitialWaterContent(value=['FC'])
    weather_file = os.path.join(weather_dir, f"{point_id}.csv")
    if not os.path.exists(weather_file): return [], []
    weather_input = pd.read_csv(weather_file)
    weather_input['Date'] = pd.to_datetime(weather_input['Date'], errors='coerce')
    yield_records = []
    seasonal_irr_records = []
    irrigation_records = []
    seasonal_twd_records = []
    total_waterdemand_records = []

    for year in range(2001, 2025):
        w_yield, w_irr_dict, w_seasonal_irr, m_seasonal_irr, m_yield, m_irr_dict , w_total_water_dict, m_total_water_dict = 0, {}, 0, 0, 0, {} ,{},{}
        w_seasonal_twd, m_seasonal_twd = 0, 0
        # get crop type
        is_wheat = attr_list[point_id][f'Wheat{year}'] == 1
        is_maize = attr_list[point_id][f'Maize{year}'] == 1
        soil_type_code = str(attr_list[point_id][f'Soiltype'])
        if soil_type_code not in soil_type_dic:
            print(f"Warning: PointID {point_id} has an unknown soil type code '{soil_type_code}' for year {year}. Skipping this point.")
            is_wheat = False
            is_maize = False
        else:
            soil_type_text = soil_type_dic.get(soil_type_code, 'unknown ')
        if is_wheat:
            w_yield, w_seasonal_irr, w_irr_dict, w_seasonal_twd, w_total_water_dict, initWC = modelingcrop('Wheat', weather_input, year, initWC,soil_type_text)
        if is_maize:
            m_yield, m_seasonal_irr, m_irr_dict, m_seasonal_twd, m_total_water_dict, initWC = modelingcrop('Maize', weather_input, year, initWC,soil_type_text)
        # save yield records
        yield_records.append({
            "PointID": point_id, "Year": year,
            "WheatYield": w_yield, "MaizeYield": m_yield,
            "x": point_coords[point_id][0], "y": point_coords[point_id][1]
        })
        # save irrigation
        seasonal_irr_records.append({
            "PointID": point_id, "Year": year,
            "WheatIrr": w_seasonal_irr, "MaizeIrr": m_seasonal_irr,
            "x": point_coords[point_id][0], "y": point_coords[point_id][1]
        })
        #save total water demand
        seasonal_twd_records.append({
            "PointID": point_id, "Year": year,
            "WheatTWD": w_seasonal_twd, "MaizeTWD": m_seasonal_twd,
            "x": point_coords[point_id][0], "y": point_coords[point_id][1]
        })

        if w_irr_dict != None and m_irr_dict != None:
            all_months = set(w_irr_dict.keys()) | set(m_irr_dict.keys())
            for m_str in all_months:
                total_irr = w_irr_dict.get(m_str, 0) + m_irr_dict.get(m_str, 0)
                if total_irr > 0:
                    irrigation_records.append({
                        "PointID": point_id,
                        "Month": m_str, # like"2024-06"
                        "Irr_mm": total_irr,
                        "x": point_coords[point_id][0], "y": point_coords[point_id][1]
                    })
        
        if w_total_water_dict != None and m_total_water_dict != None:
            all_months = set(w_total_water_dict.keys()) | set(m_total_water_dict.keys())
            for m_str in all_months:
                total_water = w_total_water_dict.get(m_str, 0) + m_total_water_dict.get(m_str, 0)
                if total_water > 0:
                    total_waterdemand_records.append({
                        "PointID": point_id,
                        "Month": m_str, # like"2024-06"
                        "TWD_mm": total_water,
                        "x": point_coords[point_id][0], "y": point_coords[point_id][1]
                    })
        

    return yield_records, seasonal_irr_records, irrigation_records, seasonal_twd_records, total_waterdemand_records

if __name__ == "__main__":
    max_workers = 12      # CPU kernels
    global initWC

    province_list = [
        "辽宁",  "河北1", "河北2", 
        "河北3", "河北4", "山西", 
        "内蒙古", "天津", "河南", "山东"
    ] 

    #分省进行计算
    for modeling_name in province_list:
        #points
        point_shp = f"./data/格网/1km分省格网/{modeling_name}.shp"
        gdf = gpd.read_file(point_shp)
        attr_list = gdf.drop(columns="geometry").to_dict(orient="records")
        point_ids = gdf.index.tolist()
        point_coords = gdf.geometry.apply(lambda geom: (geom.x, geom.y)).tolist()

        #weather data directory
        weather_dir = f"../数据预处理/Processed-Data/逐格点气象数据表/{modeling_name}/"

        # create output lists
        yield_output_records = []
        seasonal_irr_output_records = []
        irrigation_output_records = []
        seasonal_twd_records = []
        total_waterdemand_records = []#总作物需水量记录

        # mutliprocessing for each point
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, point_id in enumerate(point_ids):
                futures.append(executor.submit(run_point_simulation, i, point_id, attr_list, point_coords, weather_dir))
            for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing points"):
                y_rec, seasonal_irr_rec, irr_rec, seasonal_twd_rec, total_waterdemand_rec = f.result()
                yield_output_records.extend(y_rec)
                seasonal_irr_output_records.extend(seasonal_irr_rec)
                irrigation_output_records.extend(irr_rec)
                seasonal_twd_records.extend(seasonal_twd_rec)
                total_waterdemand_records.extend(total_waterdemand_rec)

        yield_output_df = pd.DataFrame(yield_output_records)
        seasonal_irr_output_df = pd.DataFrame(seasonal_irr_output_records)
        irrigation_output_df = pd.DataFrame(irrigation_output_records)
        seasonal_twd_df = pd.DataFrame(seasonal_twd_records)
        total_waterdemand_df = pd.DataFrame(total_waterdemand_records)

        try:
            # output yield data
            if not yield_output_df.empty:
                csv_out_path = rf"./output_soil_new/产量/{modeling_name}/"
                shp_out_path = rf"./output_soil_new/产量shp/{modeling_name}/"
                os.makedirs(csv_out_path, exist_ok=True)
                os.makedirs(shp_out_path, exist_ok=True)

                for year, year_df in yield_output_df.groupby('Year'):
                    file_name = f"{int(year)}"
                    year_df.to_csv(os.path.join(csv_out_path, f"{file_name}.csv"), index=False)
                    yield_year_gdf = gpd.GeoDataFrame(
                        year_df, 
                        geometry=gpd.points_from_xy(year_df['x'], year_df['y']), 
                        crs="EPSG:4326"
                    )

                    rename_dict = {
                        'WheatYield_tonne_per_ha': 'WheatYield',
                        'MaizeYield_tonne_per_ha': 'MaizeYield'
                    }
                    # check if the columns exist in the DataFrame before renaming
                    existing_rename = {k: v for k, v in rename_dict.items() if k in yield_year_gdf.columns}
                    yield_year_gdf = yield_year_gdf.rename(columns=existing_rename)
                    # output shp
                    yield_year_gdf.to_file(os.path.join(shp_out_path, f"yield_{file_name}.shp"))

            # output seasonal irrigation data
            if not seasonal_irr_output_df.empty:
                print("正在分年份导出作物季节灌溉数据...")
                csv_out_path = rf"./output_soil_new/种植季灌溉/{modeling_name}/"
                shp_out_path = rf"./output_soil_new/种植季灌溉shp/{modeling_name}/"
                os.makedirs(csv_out_path, exist_ok=True)
                os.makedirs(shp_out_path, exist_ok=True)

                for year, year_df in seasonal_irr_output_df.groupby('Year'):
                    file_name = f"{int(year)}"
                    year_df.to_csv(os.path.join(csv_out_path, f"{file_name}.csv"), index=False)
                    yield_year_gdf = gpd.GeoDataFrame(
                        year_df, 
                        geometry=gpd.points_from_xy(year_df['x'], year_df['y']), 
                        crs="EPSG:4326"
                    )
                    # output shp
                    yield_year_gdf.to_file(os.path.join(shp_out_path, f"seasonal_irr_{file_name}.shp"))
            
            # output monthly irrigation data
            unique_months = sorted(irrigation_output_df['Month'].unique())
            for m_str in tqdm(unique_months, desc="Exporting monthly irrigation"):
                month_data = irrigation_output_df[irrigation_output_df['Month'] == m_str]
                # outputCSV
                path_irr = rf"./output_soil_new/逐月灌溉记录/{modeling_name}/"
                os.makedirs(path_irr, exist_ok=True)
                month_data.to_csv(f"{path_irr}/{m_str}.csv", index=False)
                # outputSHP
                irr_month_gdf = gpd.GeoDataFrame(
                    month_data, 
                    geometry=gpd.points_from_xy(month_data['x'], month_data['y']), 
                    crs="EPSG:4326"
                )
                path_irr_shp = rf"./output_soil_new/逐月灌溉记录shp/{modeling_name}/"
                os.makedirs(path_irr_shp, exist_ok=True)
                irr_month_gdf.to_file(f"{path_irr_shp}/{m_str}.shp")

            # output seasonal total water demand data
            if not seasonal_twd_df.empty:
                print("正在导出种植季总需水量数据...")
                csv_out_path = rf"./output_soil_new/种植季总需水量/{modeling_name}/"
                os.makedirs(csv_out_path, exist_ok=True) 
                for year, year_df in seasonal_twd_df.groupby('Year'):
                    year_df.to_csv(os.path.join(csv_out_path, f"seasonal_twd_{int(year)}.csv"), index=False)
                    #output shp
                    seasonal_twd_gdf = gpd.GeoDataFrame(
                        seasonal_twd_df, 
                        geometry=gpd.points_from_xy(seasonal_twd_df['x'], seasonal_twd_df['y']), 
                        crs="EPSG:4326"
                    )
                    shp_out_path = rf"./output_soil_new/种植季总需水量shp/{modeling_name}/"
                    os.makedirs(shp_out_path, exist_ok=True)
                    seasonal_twd_gdf.to_file(os.path.join(shp_out_path, f"seasonal_twd_{int(year)}.shp"))

            #output monthly total water demand data
            unique_months = sorted(total_waterdemand_df['Month'].unique())
            for m_str in tqdm(unique_months, desc="Exporting monthly total water demand"):
                month_data = total_waterdemand_df[total_waterdemand_df['Month'] == m_str]
                # outputCSV
                path_total_water = rf"./output_soil_new/逐月总需水量记录/{modeling_name}/"
                os.makedirs(path_total_water, exist_ok=True)
                month_data.to_csv(f"{path_total_water}/{m_str}.csv", index=False)
                # outputSHP
                total_water_month_gdf = gpd.GeoDataFrame(
                    month_data, 
                    geometry=gpd.points_from_xy(month_data['x'], month_data['y']), 
                    crs="EPSG:4326"
                )
                path_total_water_shp = rf"./output_soil_new/逐月总需水量记录shp/{modeling_name}/"
                os.makedirs(path_total_water_shp, exist_ok=True)
                total_water_month_gdf.to_file(f"{path_total_water_shp}/{m_str}.shp")
                
        except:
            continue
        

        del yield_output_df, seasonal_irr_output_df, irrigation_output_df,yield_output_records,seasonal_irr_output_records,irrigation_output_records 
        del seasonal_twd_df, total_waterdemand_df, seasonal_twd_records, total_waterdemand_records        
        gc.collect() 
  
