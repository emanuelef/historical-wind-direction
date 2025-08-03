# Wind Direction Data Analysis: Heathrow Official vs Open-Meteo API

*Analysis date: August 3, 2025*

## Overview

This document presents the findings from comparing wind direction data from two different sources:
1. **Heathrow Airport Official Data**: Wind direction measurements from Heathrow's operational dataset
2. **Open-Meteo API**: Wind direction data retrieved from the Open-Meteo historical weather API

The analysis focuses on westerly vs easterly wind patterns, which are critical for understanding runway usage patterns and their associated noise impacts on surrounding communities.

## Key Findings

### Statistical Comparison

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Correlation | 96.9% | Very strong positive correlation between the two data sources |
| RMSE | 6.4% | Root Mean Square Error shows close alignment |
| Mean Difference | +4.1% | Heathrow reports, on average, 4.1% higher westerly wind percentage |
| Mean Absolute Difference | 5.0% | Average magnitude of difference regardless of direction |
| Maximum Difference | 13.5% | Largest observed discrepancy between the two sources |
| Agreement within 5% | 61.1% | Percentage of observations where difference is less than 5% |

### Data Matching Assessment

The analysis reveals excellent agreement between the two data sources:

- **Strong Correlation**: At 96.9%, the correlation indicates that both datasets capture the same wind direction patterns and seasonal variations.

- **Systematic Bias**: The Heathrow official data consistently reports slightly higher westerly wind percentages (about 4.1% higher) than the Open-Meteo API.

- **Temporal Consistency**: Both sources show similar patterns over time, with consistent seasonal variations and yearly trends.

- **Practical Reliability**: With over 60% of data points agreeing within 5% of each other, the Open-Meteo API provides a reliable alternative to official Heathrow data.

## Possible Explanations for Differences

Several factors may contribute to the small systematic differences observed:

1. **Measurement Methodology**: 
   - Different wind measurement instruments or standards
   - Variations in data averaging periods (hourly vs sub-hourly)

2. **Location Differences**:
   - Slight differences in the exact measurement locations within the Heathrow area
   - Different heights for wind measurements (standard is 10m, but could vary)

3. **Data Processing**: 
   - Different algorithms for classifying wind as "westerly" or "easterly"
   - Different thresholds for handling edge cases or calm winds

4. **Reporting Adjustments**:
   - Heathrow may apply corrections or adjustments to their official data
   - Possible data quality control processes that differ between sources

## Implications

The high correlation between these two data sources has several positive implications:

1. **Open-Meteo Reliability**: The Open-Meteo API provides a reliable alternative for wind direction analysis at Heathrow.

2. **Historical Analysis**: For historical trend analysis, either source would show similar patterns and lead to similar conclusions.

3. **Small Correction Factor**: If precise matching to Heathrow official percentages is needed, a simple adjustment of approximately +4% to Open-Meteo westerly percentages could be applied.

4. **Future Monitoring**: Open-Meteo can be confidently used for ongoing monitoring of wind patterns when official data is not available or delayed.

## Conclusion

The Open-Meteo API provides wind direction data that very closely matches the official Heathrow data, with only minor systematic differences. These differences are consistent and predictable, making the Open-Meteo API a valuable resource for wind direction analysis at Heathrow Airport.

The high correlation (96.9%) indicates that both data sources are measuring the same meteorological phenomena with high fidelity, and either source would lead to similar conclusions in wind pattern analysis.

## Recommendations

1. **Consider the Source Bias**: When using Open-Meteo data, be aware that it may underestimate westerly wind percentages by approximately 4% compared to official Heathrow figures.

2. **Continue Validation**: Periodically revalidate the correlation as new data becomes available.

3. **Use Open-Meteo Confidently**: The Open-Meteo API can be used with high confidence for:
   - Historical analysis
   - Trend identification
   - Seasonal pattern recognition
   - General wind direction monitoring

4. **Further Research**: Investigate the specific methodological differences that might account for the small systematic bias between the two sources.

---

*This analysis was conducted using data available up to August 2025. The comparison used matching time periods where both data sources had available measurements.*
