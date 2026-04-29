library(rsample)
library(ggplot2)
library(caTools)
library(corrplot)
library(readr)
library(Metrics)

# Read file
file.exists("C:/Users/User/Documents/Code R/Data_cleaned_for_analysis.csv")
data <- read.csv("C:/Users/User/Documents/Code R/Data_cleaned_for_analysis.csv")

# Only choose quantitative variables for analysis
numeric_data <- data[, sapply(data, is.numeric) & names(data) != "year"]

# 3.1. Pearson correlation coefficient
# For loop
maxCor = -Inf
minCor = Inf
maxX = ""
maxY = ""
minX = ""
minY = ""

maxCorList <- list("cor" = maxCor, "var1" = maxX, "var2" = maxY)
minCorList <- list("cor" = minCor, "var1" = minX, "var2" = minY)

for (i in 1 : 9)
{
  #Calculate correlation matrix, sort in descending order of correlation
  Y_name <- names(numeric_data)[i]
  cor_values <- cor_matrix[Y_name, ]
  cor_values <- sort(abs(cor_values), decreasing = TRUE)
  print(cor_values)
  
  # Choose the X variable with the highest/lowest correlation
  if (cor_values[2] >= maxCorList$cor[1])
  {
    maxCorList$cor = cor_values[2]
    maxCorList$var1 = names(cor_values)[2]
    maxCorList$var2 = names(numeric_data)[i]
  }
  
  if (cor_values[9] < minCorList$cor)
  {
    minCorList$cor = cor_values[9]
    minCorList$var1 = names(cor_values)[9]
    minCorList$var2 = names(numeric_data)[i]
  }
}

#Print results to screen
cat("Highest absolute correlation:", maxCorList$cor, "\n")
cat("Variable", maxCorList$var1, "in correlation to", maxCorList$var2, "\n")
cat("\n")

cat("Lowest absolute correlation:", minCorList$cor, "\n")
cat("Variable", minCorList$var1, "in correlation to", minCorList$var2, "\n")

# 3.2. Simple linear regression
#Split dataset
set.seed(123)

split <- initial_split(data = numeric_data, prop = 0.8, 
                       strata = sanitation_coverage_pct)
train <- training(split)
test <- testing(split)

#Build regression model
model <- lm(sanitation_coverage_pct ~ contaminant_ppm, data = train)
summary(model)


#Calculation of coefficients:
x = train$contaminant_ppm
y = train$sanitation_coverage_pct
n0 = nrow(numeric_data)
n = nrow(train)

# mean x
a = print(mean(x))
#sum xi
a = print(sum(x))
#sum xi^2
a = print(sum(x^2))
#(sum xi)^2
a = print(sum(x)**2)

#mean y
a = print(mean(y))
#sum yi
a = print(sum(y))

#sum yixi
a=print(sum(x*y))

print(format(round(a, 6), nsmall = 6))


#Plots:
#Residual and Q-Q plots
plot(model, col = "lightblue", lwd = 2)

#Histogram
hist(numeric_data$sanitation_coverage_pct, main = "",
     xlab = "sanitation_coverage_pct", col = "lightblue")

#Scatter plot and regression line
plot(numeric_data$contaminant_ppm, numeric_data$sanitation_coverage_pct, 
     xlab="contaminant_ppm", ylab="sanitation_coverage_pct",
     col="lightblue")
abline(model, col = "red", lwd = 2)
