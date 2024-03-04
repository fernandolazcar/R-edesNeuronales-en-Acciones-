---
title: "analisis de portafolio en acciones"
author: "Fernando Lazcano Cardenas"
date: "2024-02-22"
output: html_document
---


## analisis de prediccion usando redes neuronales 
## modelo de maching lerning  para pronostico de acciones 


Cargamos las librerias para nuestro proyecto 
```{r}

library(quantmod)
library(lubridate)
library(dplyr)
library(tidyr)
library(ggplot2)
library(caret)
library(neuralnet)
library(NeuralNetTools)
library(devtools)
library(PerformanceAnalytics)
library(randomForest)
library(ggthemes)
library(scales)

```

Usamos el dia de hoy para tomar la fecha de partida, tomamos la accion de nuestro interes, omitimos los datos que no tienen valor y graficamos, esto nos dara el grafico en tiempo real de las acciones de la compañia que estamos investigando.  

También podremos grabar el dos retorno diarios de la fecha de nuestro interés para ver las oscilaciones que tiene la compañía conforme el tiempo asimismo podemos ver el histograma de la frecuencia de precios que es mas repetitiva 

```{r}
hoy <- today()

ac <- getSymbols("ALFAA.MX", from = "2023-01-01", to = hoy, src = "yahoo", 
                 auto.assign = F) [,6]

ac <- na.omit(ac)

plot(ac)

#para conocer los retornos diariarios utilizamos la siguiente funcion 

r <- Return.calculate(ac)
plot(r)

#podemos ver el histograma para ver las frecuencias 
hist(r)
```




```{r}

#convertir en un dataframe y agrergar la fecha 

ac <- as.data.frame(ac)
ac$fecha <- rownames(ac)

#ver valores estadisticos 

summary (ac)

#varianza , desviacion tipica, 

var(ac)
sqrt(var(ac))

```
todo nos indica que es correcto y  que la varianza corrresponde y procesguimos

```{r}
rownames(ac) <- NULL 
names(ac) <- c("precio", "fecha")

#cambiar a formato de fecha 

ac$fecha <- as.Date(ac$fecha)
str(ac)

```
Hay que poner el ragno para hacer el pronostico el autor sugiere que la fecha del pronostico no sea mayor a 30 dias si hay mucha informacion quiza 60 dias esta bien, en esto hacemos otro dataframe para inciar con el pronostico, esto se debe concidir con el nombre de las columnas 

```{r}


rf = (hoy + 1) : (hoy + 30)
precio = as.numeric(NA)
rf = as.data.frame(cbind(precio,rf))
rf$fecha = as.Date(rf$rf)
rf$rf = NULL


#se une 
ac <- rbind(ac,rf) 

#primero vamos a crear un duplicado de la acciones / fecha 

ac$fecha_dup = ac$fecha
ac <- ac %>% separate(fecha, c("año", "mes", "dia")) 
str(ac)
ac$año = as.numeric(ac$año)
ac$mes = as.numeric(ac$mes)
ac$dia = as.numeric(ac$dia)

#escalado de datos / no olvidar sembrar la semilla de aleatoriedad 
set.seed(1996)
ac.sc <- as.data.frame(cbind(ac$precio,ac$fecha_dup, scale(ac[,c(2:4)])))
names(ac.sc)[1]= "precio"
names(ac.sc)[2]= "fecha"
ac.sc$fecha = as.Date(ac.sc$fecha)

```
Vamos a empezar con los datos aleatorios que se tomaron de la data de acciones y se filtraton en todas las fechas, que fueran de hoy y adicional con na.omit, no tomando en cuenta todas las filas que aparecen con NA.

Hay que considera que los datos son suceptibles a los datos faltantes. 

```{r}
#entrenomiento
train = createDataPartition (na.omit (subset(ac, ac$fecha_dup < today()))$precio, 
                             p = 0.7, list = F)


#proyeccion de los datos / se escala 

test = rbind (ac[-train,], subset(ac, ac$fecha_dup >= today()))
test.sc = as.data.frame(cbind(test$precio, test$fecha_dup, scale (test [, c(2,3,4)])))
names(test.sc)[1] = "precio"
names(test.sc)[2] = "fecha"
test.sc$fecha = as.Date(test.sc$fecha)
```

Esto es bastante poderoso pero tiene desvetajas las cuales son  :
 1 requiere escalo de datos ( osea las fechas )
 2 no sirve para tiempo real 
 3 es muy sensible a las variables 

El modelo tiene que converger para los valores sean adecuado ( osea stepmax sea alto) alinar los output sirve para vietar que los valores de salida y pierden la capacidad de ser interpretados / hidden corresponde a las dos neuronas

que son h1 y h2

```{r}

mod = neuralnet(formula = precio ~ año + mes + dia, data = ac.sc [train, ], hidden = 2,
                threshold = 0.01, stepmax = 1e+08,  rep = 1, linear.output = T )


```

Se genera el grafica, se agregaran mas neuronas cuando hay muchos datos de salida si queremos datos muy preciso  y la funcion del grandiente desendiente para que los errores se reduzcan mas ; el grosor de la lineas nos dice en que propocion o magnitud es relevante cada una de las variables de netrada ( las Bs significas bayas)
casi siempre son signficativos que apoyana a la prediccion.


```{r}
plotnet(mod)

pred <- compute(mod, test.sc)
```

Observemos la prediccion en los primeros valores parece correecto pero despues de la fila 7 no me parece tan bueno porque sube a 20 dolares 

```{r}
datos = cbind(pred$net, test.sc )

#veamos los errores 

er = RMSE (datos$precio, datos $`pred$net`, na.rm = T)
erp = er/ datos[datos$fecha == max(na.omit(datos)$fecha),]$precio


ggplot() +geom_line(data = datos, aes ( x =fecha, y =precio), color = "black")+
  geom_line (data= datos, aes (x= fecha , y = `pred$net`), color = "red")

```
modelos a investigar Random Forest, Extreme Radio Boost  ; no requieren escalado de datos y los modelo Ramdom Forest es un arboles de decisiones y toma mejores desicione

Con este modelo es inesesario usar el escalado de datos, el numero de datos que da es de 500 arboles para fines practicos usaremos 100, el sobre usar el numero de arboles provocara un sobre porcesamiento y sobre ajuste 


```{r}

mod_rf = randomForest(precio ~ año + mes + dia, data = ac [train, ], 
                      type= "regression",  ntree = 100 )


#prediccion 

pred_rf = predict(mod_rf, test)
datos_rf = cbind(pred_rf, test)



er_rf = RMSE (datos_rf$precio, datos_rf$pred_rf, na.rm = T)
erx_rf = er_rf / datos_rf[datos_rf$fecha_dup == max (na.omit(datos_rf)$fecha_dup),]$precio

#grafico
ggplot() +geom_line(data = datos_rf, aes ( x =fecha_dup, y =precio), color = "black")+
  geom_line (data= datos_rf, aes (x= fecha_dup , y = `pred_rf`), color = "red")

```


ejercicio realizado por Fernando Lazcano Cardenas

fuentes : DO Analytics "https://www.youtube.com/watch?v=RBr3yRdE_LA&ab_channel=DOAnalytics"


https://rpubs.com/yevonnael/intro-portfolio-analysis

