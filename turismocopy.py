#!/usr/bin/env python
# coding: utf-8

# ## Sesion Pyspark

# In[1]:


from pyspark.sql import SparkSession
from pyspark import SparkContext

SpSession = SparkSession \
    .builder \
    .appName("Pyspark Turist Canary") \
    .getOrCreate()

SpSession.sparkContext.setLogLevel("ERROR")


# In[2]:


from pyspark.sql.functions import *
import pandas as pd


# ## Carga de Datos

# In[3]:


df_pd_hotel = pd.read_csv("https://datos.canarias.es/api/estadisticas/statistical-resources/v1.0/datasets/ISTAC/C00065A_000048/1.15.csv")
df_sp_hotel = SpSession.createDataFrame(df_pd_hotel)


# In[4]:


df_sp_hotel.show(5, truncate=False)


# In[5]:


df_pd_type = pd.read_csv("https://datos.canarias.es/api/estadisticas/statistical-resources/v1.0/datasets/ISTAC/E16028B_000013/1.47.csv")
df_sp_type = SpSession.createDataFrame(df_pd_type)


# In[6]:


df_sp_type.show(5, truncate=False)


# In[7]:


df_pd = pd.read_csv("https://datos.canarias.es/api/estadisticas/statistical-resources/v1.0/datasets/ISTAC/E16028B_000011/~latest.csv")
df_sp = SpSession.createDataFrame(df_pd)


# In[8]:


df_sp.show(5, truncate=False)


# ## Exploración y Limpieza

# In[9]:


df_sp = df_sp.na.drop(subset=["OBS_VALUE"])


# In[10]:


df_sp_type = df_sp_type.na.drop(subset=["OBS_VALUE"])


# In[11]:


df_sp_hotel = df_sp_hotel.na.drop(subset=["OBS_VALUE"])


# In[12]:


df_sp.describe().show()


# In[13]:


df_sp = df_sp.filter(col("LUGAR_RESIDENCIA#es") != "Total")


# In[14]:


df_sp = df_sp.filter(col("MEDIDAS#es") == "Turistas")


# In[15]:


df_sp = df_sp.withColumn("mes", split(col("TIME_PERIOD#es"), "/").getItem(0)) \
       .withColumn("año", split(col("TIME_PERIOD#es"), "/").getItem(1))


# In[16]:


df_sp.select("MEDIDAS#es").distinct().show(20)


# In[17]:


df_sp_type.describe().show()


# In[18]:


df_sp_type = df_sp_type.filter(col("MEDIDAS#es") == "Turistas")


# In[19]:


df_sp_type = df_sp_type.filter(col("LUGAR_RESIDENCIA#es") != "Total")


# In[20]:


df_sp_type = df_sp_type.filter(col("ALOJAMIENTO_TURISTICO_TIPO#es") != "Total")


# In[21]:


df_sp_type = df_sp_type.withColumn("mes", split(col("TIME_PERIOD#es"), "/").getItem(0)) \
       .withColumn("año", split(col("TIME_PERIOD#es"), "/").getItem(1))


# In[22]:


df_sp_hotel.describe().show()


# In[23]:


df_sp_hotel = df_sp_hotel.withColumn(
    "isla#es",
    when(col("TERRITORIO#es").isin(
        "Adeje", "Arona", "Granadilla de Abona", "Guía de Isora",
        "Puerto de la Cruz", "San Miguel de Abona", "San Cristóbal de La Laguna",
        "Santa Cruz de Tenerife", "Santiago del Teide", "Tenerife", "Resto de Tenerife"
    ), "Tenerife")
    .when(col("TERRITORIO#es").isin(
        "Las Palmas de Gran Canaria", "Mogán", "San Bartolomé de Tirajana", "Resto de Gran Canaria", "Gran Canaria"
    ), "Gran Canaria")
    .when(col("TERRITORIO#es").isin(
        "Tías", "Teguise", "Yaiza", "Arrecife", "San Bartolomé", "Lanzarote"
    ), "Lanzarote")
    .when(col("TERRITORIO#es").isin(
        "Pájara", "Tuineje", "Antigua", "La Oliva", "Fuerteventura"
    ), "Fuerteventura")
    .when(col("TERRITORIO#es").isin(
        "Los Llanos de Aridane", "Breña Baja", "La Palma"
    ), "La Palma")
    .when(col("TERRITORIO#es").isin(
        "San Sebastián de La Gomera", "La Gomera"
    ), "La Gomera")
    .when(col("TERRITORIO#es") == "Canarias", "Canarias")
    .otherwise("Desconocido")
)


# In[24]:


df_sp_hotel.filter(col("isla#es") == "Canarias").count()


# In[25]:


df_sp_hotel = df_sp_hotel.filter(col("isla#es") != "Canarias")


# In[26]:


df_sp_hotel.select("MEDIDAS#es").distinct().show(40, truncate=False)


# In[27]:


df_sp.dtypes
#df_sp_type.dtypes
#df_sp_hotel.dtypes


# In[28]:


df_sp.show(5, truncate=False)
df_sp_type.show(5, truncate=False)
df_sp_hotel.show(5, truncate=False)


# ## Querys Simple
# 

# ### Analisis de tendencias temporales

# In[29]:


evolucion_anual = df_sp.groupBy("año") \
    .agg(sum("OBS_VALUE").alias("total_turistas")) \
    .orderBy("año")

evolucion_anual.show()


# In[30]:


estacionalidad = df_sp.groupBy("mes") \
    .agg(avg("OBS_VALUE").alias("promedio_turistas")) \
    .orderBy("mes")

print("\n * ESTACIONALIDAD PROMEDIO *")
estacionalidad.show()


# ### Origen del turista

# In[31]:


top_paises = df_sp.groupBy("LUGAR_RESIDENCIA#es") \
    .agg(sum("OBS_VALUE").alias("total_turistas")) \
    .orderBy(desc("total_turistas")) \
    .limit(10)

top_paises.show(10, truncate=False)


# In[32]:


principales_mercados = df_sp.filter(col("LUGAR_RESIDENCIA#es").isin(["Reino Unido", "Alemania", "España (excluida Canarias)", "Países Nórdicos", "Francia", "Países Bajos"])) \
    .groupBy("año", "LUGAR_RESIDENCIA#es") \
    .agg(sum("OBS_VALUE").alias("total_turistas")) \
    .orderBy("año", "LUGAR_RESIDENCIA#es")

print("\n * EVOLUCIÓN PRINCIPALES MERCADOS * ")
principales_mercados.show(truncate=False)


# ### Analisis por islas

# In[33]:


from pyspark.sql.window import Window

distribucion_islas = df_sp.groupBy("TERRITORIO#es") \
    .agg(sum("OBS_VALUE").alias("total_turistas")) \
    .withColumn("porcentaje", 
                round(col("total_turistas") * 100.0 / sum("total_turistas").over(Window.partitionBy()), 2)) \
    .orderBy(desc("total_turistas"))

distribucion_islas.show()


# In[34]:


window_isla = Window.partitionBy("TERRITORIO#es").orderBy("año")

crecimiento_islas = df_sp.filter(col("año") >= 2019) \
    .groupBy("TERRITORIO#es", "año") \
    .agg(sum("OBS_VALUE").alias("turistas_año")) \
    .withColumn("turistas_año_anterior", lag("turistas_año").over(window_isla)) \
    .withColumn("crecimiento_porcentual", 
                round(((col("turistas_año") - col("turistas_año_anterior")) / col("turistas_año_anterior")) * 100, 2)) \
    .filter(col("crecimiento_porcentual").isNotNull()) \
    .orderBy("TERRITORIO#es", "año")

print("\n* CRECIMIENTO ANUAL POR ISLA (2019-2025) *")
crecimiento_islas.show(truncate=False)


# ### Tipo de alojamiento
# 

# In[35]:


alojamiento_preferencias = df_sp_type.groupBy("ALOJAMIENTO_TURISTICO_TIPO#es") \
    .agg(sum("OBS_VALUE").alias("total_turistas")) \
    .withColumn("porcentaje", 
                round(col("total_turistas") * 100.0 / sum("total_turistas").over(Window.partitionBy()), 2)) \
    .orderBy(desc("total_turistas"))

alojamiento_preferencias.show(truncate=False)


# In[36]:


alojamiento_por_pais = df_sp_type.groupBy("LUGAR_RESIDENCIA#es", "ALOJAMIENTO_TURISTICO_TIPO#es") \
    .agg(sum("OBS_VALUE").alias("turistas")) \
    .withColumn("total_pais", sum("turistas").over(Window.partitionBy("LUGAR_RESIDENCIA#es"))) \
    .withColumn("porcentaje", round(col("turistas") * 100.0 / col("total_pais"), 2)) \
    .filter(col("total_pais") > 100000) \
    .orderBy("LUGAR_RESIDENCIA#es", desc("porcentaje"))

print("\n * PREFERENCIAS DE ALOJAMIENTO POR PAÍS *")
alojamiento_por_pais.show(truncate=False)


# ### Eventos especiales

# In[37]:


ranking_eventos = df_sp_hotel.groupBy("EVENTO#es").agg(
    avg(when(col("MEDIDAS#es") == "Pernoctaciones", col("OBS_VALUE"))).alias("promedio_pernoctaciones"),
    avg(when(col("MEDIDAS#es") == "Viajeros entrados", col("OBS_VALUE"))).alias("promedio_viajeros_entrados"),
    avg(when(col("MEDIDAS#es") == "Viajeros alojados", col("OBS_VALUE"))).alias("promedio_viajeros_alojados")
).orderBy(desc("promedio_pernoctaciones"))

ranking_eventos.show(truncate=False)


# In[38]:


evolucion_eventos = df_sp_hotel.groupBy("TIME_PERIOD#es", "EVENTO#es") \
    .agg(sum(when(col("MEDIDAS#es") == "Pernoctaciones", col("OBS_VALUE"))).alias("total_pernoctaciones")) \
    .orderBy("TIME_PERIOD#es", "EVENTO#es")

print("\n* EVOLUCIÓN DE EVENTOS ESPECIALES*")
evolucion_eventos.show(truncate=False)


# ## Querys Avanzadas

# ### Covid-19

# In[39]:


impacto_covid = df_sp.filter(col("año").isin([2019, 2020, 2021])) \
    .groupBy("año", "TERRITORIO#es") \
    .agg(sum("OBS_VALUE").alias("total_turistas")) \
    .withColumn("año_base", when(col("año") == 2019, col("total_turistas")).otherwise(None)) \
    .withColumn("año_base_filled", 
                last("año_base", ignorenulls=True).over(Window.partitionBy("TERRITORIO#es").orderBy("año"))) \
    .withColumn("variacion_covid", 
                round(((col("total_turistas") - col("año_base_filled")) / col("año_base_filled")) * 100, 2)) \
    .filter(col("año") != 2019) \
    .select("año", "TERRITORIO#es", "total_turistas", "variacion_covid") \
    .orderBy("TERRITORIO#es", "año")

print("\n * IMPACTO COVID-19 *")
impacto_covid.show(truncate=False)


# In[40]:


recuperacion_post_covid = df_sp.filter(col("año").isin([2019, 2022, 2023, 2024])) \
    .groupBy("año", "TERRITORIO#es") \
    .agg(sum("OBS_VALUE").alias("total_turistas")) \
    .withColumn("turistas_2019", 
                when(col("año") == 2019, col("total_turistas")).otherwise(None)) \
    .withColumn("turistas_2019_filled", 
                last("turistas_2019", ignorenulls=True).over(Window.partitionBy("TERRITORIO#es").orderBy("año"))) \
    .withColumn("porcentaje_recuperacion", 
                round((col("total_turistas") / col("turistas_2019_filled")) * 100, 2)) \
    .filter(col("año") != 2019) \
    .select("año", "TERRITORIO#es", "total_turistas", "porcentaje_recuperacion") \
    .orderBy("TERRITORIO#es", "año")

print("\n  *RECUPERACIÓN POST-COVID *")
recuperacion_post_covid.show()


# ### Correlaciones

# In[41]:


df_eventos_pivot = df_sp_hotel.groupBy("TIME_PERIOD#es", "EVENTO#es", "isla#es") \
    .pivot("MEDIDAS#es") \
    .sum("OBS_VALUE") \
    .withColumnRenamed("Pernoctaciones", "pernoctaciones") \
    .withColumnRenamed("Viajeros entrados", "viajeros_entrados") \
    .withColumnRenamed("Viajeros alojados", "viajeros_alojados")

print("\n=== VISTA PREVIA DE DATOS PIVOTEADOS ===")
df_eventos_pivot.show(5)


# In[42]:


correlacion_eventos = df_eventos_pivot.select(
    corr("pernoctaciones", "viajeros_entrados").alias("corr_pernoc_viajeros"),
    corr("pernoctaciones", "viajeros_alojados").alias("corr_pernoc_alojados"),
    corr("viajeros_entrados", "viajeros_alojados").alias("corr_viajeros")
)

print("\n* CORRELACIONES EN EVENTOS ESPECIALES*")
correlacion_eventos.show()


# In[43]:


correlaciones_por_isla = df_eventos_pivot.groupBy("isla#es") \
    .agg(
        corr("pernoctaciones", "viajeros_entrados").alias("corr_pernoc_viajeros"),
        corr("pernoctaciones", "viajeros_alojados").alias("corr_pernoc_alojados"),
        corr("viajeros_entrados", "viajeros_alojados").alias("corr_viajeros")
    ) \
    .orderBy("isla#es")

print("\n *CORRELACIONES POR ISLA *")
correlaciones_por_isla.show()


# In[44]:


correlaciones_por_evento = df_eventos_pivot.groupBy("EVENTO#es") \
    .agg(
        corr("pernoctaciones", "viajeros_entrados").alias("corr_pernoc_viajeros"),
        corr("pernoctaciones", "viajeros_alojados").alias("corr_pernoc_alojados"),
        corr("viajeros_entrados", "viajeros_alojados").alias("corr_viajeros"),
        count("*").alias("num_observaciones")
    ) \
    .filter(col("num_observaciones") >= 30) \
    .orderBy("EVENTO#es")

print("\n* CORRELACIONES POR EVENTO *")
correlaciones_por_evento.show(truncate=False)


# In[45]:


correlaciones_anuales = df_eventos_pivot.groupBy("TIME_PERIOD#es") \
    .agg(
        corr("pernoctaciones", "viajeros_entrados").alias("corr_pernoc_viajeros"),
        corr("pernoctaciones", "viajeros_alojados").alias("corr_pernoc_alojados"),
        corr("viajeros_entrados", "viajeros_alojados").alias("corr_viajeros")
    ) \
    .orderBy("TIME_PERIOD#es")

print("\n  *EVOLUCIÓN DE CORRELACIONES POR AÑO*")
correlaciones_anuales.show(truncate=False)


# ## INSIGHTS Especificos

# In[46]:


mejor_mes_isla = df_sp.groupBy("TERRITORIO#es", "mes") \
    .agg(avg("OBS_VALUE").alias("promedio_turistas")) \
    .withColumn("rank", 
                row_number().over(Window.partitionBy("TERRITORIO#es").orderBy(desc("promedio_turistas")))) \
    .filter(col("rank") == 1) \
    .select("TERRITORIO#es", "mes", "promedio_turistas") \
    .orderBy("TERRITORIO#es")

print("\n=== MEJOR MES PARA CADA ISLA ===")
mejor_mes_isla.show(truncate=False)


# In[47]:


crecimiento_paises = df_sp.filter(col("año").isin([2021, 2024])) \
    .groupBy("LUGAR_RESIDENCIA#es", "año") \
    .agg(sum("OBS_VALUE").alias("total_turistas")) \
    .groupBy("LUGAR_RESIDENCIA#es") \
    .pivot("año") \
    .sum("total_turistas") \
    .withColumn("crecimiento_porcentual", 
                round(((col("2024") - col("2021")) / col("2021")) * 100, 2)) \
    .filter(col("2021") > 10000) \
    .orderBy(desc("crecimiento_porcentual")) \
    .limit(10)

print("\n=== PAÍSES CON MAYOR CRECIMIENTO (2021-2024) ===")
crecimiento_paises.show(truncate=False)

