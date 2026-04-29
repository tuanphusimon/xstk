# ============================================================
# MT2013 - Probability and Statistics
# Part 2: Data Cleaning / Preprocessing
# Dataset: Data_raw.csv
# ============================================================

rm(list = ls())
options(stringsAsFactors = FALSE, scipen = 999)

# 3.1 Read data

# Read the raw dataset from csv file
raw_df <- read.csv(
  "Data_raw.csv",
  check.names = FALSE,
  na.strings = c("", "NA", "N/A")
)

# Display general information about the raw dataset
dim(raw_df)
str(raw_df)
head(raw_df)
summary(raw_df)

# View the entire raw dataset in RStudio
View(raw_df)

# Check duplicated rows
duplicate_rows <- sum(duplicated(raw_df))
duplicate_rows

# 3.2 Extract a subset of important variables

selected_cols <- c(
  "Country",
  "Region",
  "Year",
  "Water Source Type",
  "Water Treatment Method",
  "Contaminant Level (ppm)",
  "pH Level",
  "Bacteria Count (CFU/mL)",
  "Access to Clean Water (% of Population)",
  "Diarrheal Cases per 100,000 people",
  "Infant Mortality Rate (per 1,000 live births)",
  "GDP per Capita (USD)",
  "Healthcare Access Index (0-100)",
  "Sanitation Coverage (% of Population)"
)

analysis_df <- raw_df[, selected_cols]

# Display the subset of selected variables
dim(analysis_df)
head(analysis_df)
View(analysis_df)

# 3.3 Handle data format

# Rename variables for easier analysis in R
colnames(analysis_df) <- c(
  "country",
  "region",
  "year",
  "water_source_type",
  "water_treatment_method",
  "contaminant_ppm",
  "ph_level",
  "bacteria_count",
  "access_clean_water_pct",
  "diarrheal_cases_per_100k",
  "infant_mortality_rate",
  "gdp_per_capita_usd",
  "healthcare_access_index",
  "sanitation_coverage_pct"
)

# Identify character and numeric columns
char_cols <- c(
  "country",
  "region",
  "water_source_type",
  "water_treatment_method"
)

num_cols <- c(
  "year",
  "contaminant_ppm",
  "ph_level",
  "bacteria_count",
  "access_clean_water_pct",
  "diarrheal_cases_per_100k",
  "infant_mortality_rate",
  "gdp_per_capita_usd",
  "healthcare_access_index",
  "sanitation_coverage_pct"
)

# Remove leading and trailing spaces in character columns
analysis_df[char_cols] <- lapply(analysis_df[char_cols], function(x) {
  x <- trimws(x)
  x[x == ""] <- NA
  return(x)
})

# Convert numeric variables to numeric type explicitly
analysis_df[num_cols] <- lapply(analysis_df[num_cols], as.numeric)

# Ensure year is stored as integer
analysis_df$year <- as.integer(analysis_df$year)

# Check data structure after formatting
str(analysis_df)
summary(analysis_df)
View(analysis_df)

# 3.4 Handle missing data

# Count the number of missing values in each variable
missing_count_before <- sapply(analysis_df, function(x) sum(is.na(x)))

# Compute the percentage of missing values in each variable
missing_pct_before <- round(sapply(analysis_df, function(x) mean(is.na(x)) * 100), 2)

# Combine missing statistics into one table
missing_summary_before <- data.frame(
  variable = names(missing_count_before),
  missing_count = as.integer(missing_count_before),
  missing_pct = as.numeric(missing_pct_before)
)

# Display missing-value summary before treatment
missing_summary_before
View(missing_summary_before)

# Check category frequencies for water_treatment_method before imputation
table(analysis_df$water_treatment_method, useNA = "ifany")

# Replace missing values in water_treatment_method with "Unknown"
analysis_df$water_treatment_method[is.na(analysis_df$water_treatment_method)] <- "Unknown"

# Check missing values again after treatment
missing_count_after <- sapply(analysis_df, function(x) sum(is.na(x)))
missing_pct_after <- round(sapply(analysis_df, function(x) mean(is.na(x)) * 100), 2)

missing_summary_after <- data.frame(
  variable = names(missing_count_after),
  missing_count = as.integer(missing_count_after),
  missing_pct = as.numeric(missing_pct_after)
)

# Display missing-value summary after treatment
missing_summary_after
View(missing_summary_after)

# Check category frequencies for water_treatment_method after imputation
table(analysis_df$water_treatment_method, useNA = "ifany")

# 3.5 Detect outliers using the IQR rule

iqr_summary <- data.frame(
  variable = character(),
  q1 = numeric(),
  q3 = numeric(),
  iqr = numeric(),
  lower_bound = numeric(),
  upper_bound = numeric(),
  n_outliers = integer(),
  stringsAsFactors = FALSE
)

for (col in num_cols) {
  q1_value <- unname(quantile(analysis_df[[col]], 0.25, na.rm = TRUE))
  q3_value <- unname(quantile(analysis_df[[col]], 0.75, na.rm = TRUE))
  iqr_value <- IQR(analysis_df[[col]], na.rm = TRUE)
  
  lower_value <- q1_value - 1.5 * iqr_value
  upper_value <- q3_value + 1.5 * iqr_value
  
  n_outlier_value <- sum(
    analysis_df[[col]] < lower_value | analysis_df[[col]] > upper_value,
    na.rm = TRUE
  )
  
  iqr_summary <- rbind(
    iqr_summary,
    data.frame(
      variable = col,
      q1 = q1_value,
      q3 = q3_value,
      iqr = iqr_value,
      lower_bound = lower_value,
      upper_bound = upper_value,
      n_outliers = n_outlier_value
    )
  )
}

# Display IQR outlier summary
iqr_summary
View(iqr_summary)

# Variables flagged by the IQR rule
iqr_flagged <- iqr_summary[iqr_summary$n_outliers > 0, ]
iqr_flagged


# 3.6 Final cleaned dataset=

# Because IQR results show no outliers,
# all observations are retained in the cleaned dataset.
clean_df <- analysis_df

# Create a cleaned version of the full raw dataset as well
full_clean_df <- raw_df

# Trim spaces in all character columns of the full dataset
full_clean_df[] <- lapply(full_clean_df, function(x) {
  if (is.character(x)) {
    x <- trimws(x)
    x[x == ""] <- NA
  }
  return(x)
})

# Replace missing values in the original water treatment column (in case it has unknown value)
full_clean_df$`Water Treatment Method`[
  is.na(full_clean_df$`Water Treatment Method`)
] <- "Unknown"

# Display cleaned data
dim(clean_df)
head(clean_df)
summary(clean_df)
View(clean_df)

# 3.7 Export outputs

write.csv(clean_df, "Data_cleaned_for_analysis.csv", row.names = FALSE)
write.csv(full_clean_df, "Data_cleaned_full.csv", row.names = FALSE)
write.csv(missing_summary_before, "Missing_summary_before.csv", row.names = FALSE)
write.csv(missing_summary_after, "Missing_summary_after.csv", row.names = FALSE)
write.csv(iqr_summary, "IQR_outlier_summary.csv", row.names = FALSE)

cat("Raw dataset dimensions:", dim(raw_df)[1], "rows and", dim(raw_df)[2], "columns\n")
cat("Analysis dataset dimensions:", dim(clean_df)[1], "rows and", dim(clean_df)[2], "columns\n")
cat("Duplicated rows in raw data:", duplicate_rows, "\n")
cat("Number of variables with IQR outliers:", nrow(iqr_flagged), "\n")
cat("Files exported:\n")
cat("- Data_cleaned_for_analysis.csv\n")
cat("- Data_cleaned_full.csv\n")
cat("- Missing_summary_before.csv\n")
cat("- Missing_summary_after.csv\n")
cat("- IQR_outlier_summary.csv\n")

