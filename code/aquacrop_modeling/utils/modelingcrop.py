from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent,IrrigationManagement
import numpy as np  
from datetime import datetime
import pandas as pd
import gc
# basic irrigation management settings
irr_mngt = IrrigationManagement(
    irrigation_method=1,
    SMT=[95, 90, 85, 80],
    MaxIrr=15
)

def linear_param(year, y0, y1, p0, p1):
    return p0 + (p1 - p0) * (year - y0) / (y1 - y0)

def to_float(x):
    if isinstance(x, str):
        return float(x.strip("[]"))
    return float(x)

#执行模拟
def modelingcrop(croptype,weather_input,year,initWC,soil_type_text):
    soil_set= Soil(soil_type_text)#set soil type
    if croptype == 'Wheat':
        modeling_start = datetime(year-1, 10, 1)
        modeling_end = datetime(year, 9, 30)
        wheat_start_date = '10/1'
        wheat_end_date = '9/30'
        HI0 = linear_param(
            year,
            y0=2001, y1=2024,
            p0=0.25, p1=0.36
        )
        Zmax = linear_param(
            year,
            y0=2001, y1=2024,
            p0=1.35, p1=1.60
        )
        crop_type_set = Crop('WheatGDD', planting_date=wheat_start_date, harvest_date=wheat_end_date,HI0=HI0,Zmax=Zmax)
    elif croptype == 'Maize':
        modeling_start = datetime(year, 6, 1)
        modeling_end = datetime(year, 12, 30) if year == 2024 else datetime(year + 1, 5, 31)
        maize_start_date = '06/15'
        maize_end_date = '12/30'
        HI0 = linear_param(
            year,
            y0=2001, y1=2024,
            p0=0.18, p1=0.25
        )
        Zmax = linear_param(
            year,
            y0=2001, y1=2024,
            p0=1.35, p1=1.60
        )
        crop_type_set = Crop('MaizeGDD', planting_date=maize_start_date, harvest_date=maize_end_date,HI0=HI0,Zmax=Zmax)
    # weather data for the modeling period
    modeling_weather_input = weather_input[(weather_input['Date'] >= modeling_start) & (weather_input['Date'] <= modeling_end)]
    modeling_start_time = modeling_start.strftime("%Y/%m/%d")
    modeling_end_time = modeling_end.strftime("%Y/%m/%d")
    # run model
    try:
        crop_model = AquaCropModel(sim_start_time = modeling_start_time,
                            sim_end_time = modeling_end_time,
                            weather_df = modeling_weather_input,
                            soil = soil_set,
                            crop = crop_type_set,
                            initial_water_content= initWC,
                            irrigation_management= irr_mngt) # create model
        crop_model.run_model(till_termination=True)
        out = crop_model._outputs
        
        # get the final yield and seasonal irrigation amount
        if hasattr(out, 'final_stats') and not out.final_stats.empty:
            y = float(out.final_stats['Fresh yield (tonne/ha)'].iloc[-1])
            # get the seasonal irrigation amount and monthly irrigation amount
            seasonal_irr = float(out.final_stats['Seasonal irrigation (mm)'].iloc[-1])
            irr_df = out.water_flux[['IrrDay']].copy()
            irr_df['IrrDay'] = irr_df['IrrDay'].apply(lambda x: float(str(x).strip("[]")))
            irr_df.index = pd.to_datetime(modeling_start) + pd.to_timedelta(irr_df.index, unit='D')
            irr_monthly = irr_df['IrrDay'].resample('ME').sum()
            irr_dict = {k.strftime('%Y-%m'): float(v) for k, v in irr_monthly.items()}
            
            # get the seasonal total water demand and monthly total water demand
            soil_evaporation = out.water_flux[["Es"]]
            crop_transpiration = out.water_flux[["Tr"]]
            soil_evaporation.index = pd.to_datetime(modeling_start) + pd.to_timedelta(soil_evaporation.index, unit='D')
            crop_transpiration.index = pd.to_datetime(modeling_start) + pd.to_timedelta(crop_transpiration.index, unit='D')
            soil_evaporation_monthly = soil_evaporation['Es'].resample('ME').sum()
            crop_transpiration_monthly = crop_transpiration['Tr'].resample('ME').sum()
            total_water_monthly = soil_evaporation_monthly + crop_transpiration_monthly
            total_water_dict = {k.strftime('%Y-%m'): float(v) for k, v in total_water_monthly.items()}
            # calculate seasonal total water demand
            seasonal_twd = total_water_monthly.sum()


            # get the last water content of each layer and set it as the initial water content for the next year
            valid_storage = out.water_storage[out.water_storage['time_step_counter'] > 0]
            th_columns = [c for c in out.water_storage.columns if c.startswith('th')]
            last_th = valid_storage.iloc[-1][th_columns].values.flatten().tolist()
            depths = np.linspace(2.4/12, 2.4, 12).tolist() 
            next_init_wc = InitialWaterContent(
                wc_type='Num', 
                value=last_th, 
                depth_layer=depths
            )
            del crop_model,out,irr_df
            gc.collect() 

            return y, seasonal_irr,irr_dict, seasonal_twd, total_water_dict,next_init_wc
        else:
            return 0, 0, None, 0, None, initWC
    except Exception as e:
        # print(f"Error in modeling {croptype} for year {year}: {e}")
        return 0, 0, None, 0, None, initWC
    
    