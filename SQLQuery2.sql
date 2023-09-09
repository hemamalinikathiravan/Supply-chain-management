
select * from dbo.pallets$;


CREATE TABLE dbo.newdata
 AS
 select  distinct * from dbo.pallets$
select * from dbo.newdata$;

-- Calculate the mean for the column
-- Calculate the mean
SELECT AVG(U_frt) AS mean_value
FROM dbo.newdata$

-- Update missing values with the mean
UPDATE dbo.newdata$
SET U_frt = (SELECT AVG(U_frt) FROM dbo.newdata$)
WHERE U_frt IS NULL;

SELECT AVG(quantity) AS mean_value
FROM dbo.newdata$;

UPDATE dbo.newdata$
SET quantity = (SELECT AVG(quantity) FROM dbo.newdata$)
WHERE quantity IS NULL;

SELECT AVG(rate) AS mean_value
FROM dbo.newdata$

UPDATE dbo.newdata$
SET rate = (SELECT AVG(rate) FROM dbo.newdata$)
WHERE rate IS NULL;

/*type casting the nvarchar to float to perform missing values*/


UPDATE dbo.newdata$
SET u_grnno = (SELECT ISNULL(u_grnno, 0 )) FROM dbo.newdata$

SELECT [u_grnno], AVG(CAST(REPLACE([u_grnno],',','.') as DECIMAL(9,2))) AS [u_grnno1]
FROM dbo.newdata$
GROUP BY [u_grnno]

SELECT AVG(u_grnno) AS mean_value
FROM dbo.newdata$;

UPDATE dbo.newdata$
SET u_grnno = (SELECT AVG(u_grnno) FROM dbo.newdata$)
WHERE u_grnno IS NULL;



SELECT AVG(LoadingUnloading) AS mean_value
FROM dbo.newdata$;

UPDATE dbo.newdata$
SET LoadingUnloading = (SELECT AVG(LoadingUnloading) FROM dbo.newdata$)
WHERE LoadingUnloading IS NULL;

ALTER TABLE dbo.newdata$
DROP COLUMN detention;
ALTER TABLE dbo.newdata$
DROP COLUMN customerType;
ALTER TABLE dbo.newdata$
DROP COLUMN KITITEM;
ALTER TABLE dbo.newdata$
DROP COLUMN U_TRINPD;
 
 select * from dbo.newdata$


UPDATE dbo.newdata$
SET u_assetclass = 'Wooden Pallet' -- Replace with the desired value
WHERE u_assetclass IS NULL;


/* Handling Outliers */

/* find Outlier by this easy method */

-- Calculate the first quartile (Q1), third quartile (Q3), and interquartile range (IQR)

DECLARE @Q1 FLOAT, @Q3 FLOAT, @IQR FLOAT;

-- Calculate the total number of rows in the table
DECLARE @TotalRows INT;
SELECT @TotalRows = COUNT(*) FROM dbo.newdata$;

-- Calculate the quartiles
SELECT 
    @Q1 = MIN(U_frt) OVER (ORDER BY U_frt ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),
    @Q3 = MAX(U_frt) OVER (ORDER BY U_frt ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING),
    @IQR = @Q3 - @Q1
FROM dbo.newdata$
WHERE NTILE(4) OVER (ORDER BY U_frt) = 2; -- Q2 (median)

-- Identify outliers
SELECT *
FROM U_frt
WHERE U_frt < @Q1 - 1.5 * @IQR OR U_frt > @Q3 + 1.5 * @IQR;


/*  way to find the zero_variance */
SELECT count(*) as zero_variance,postingdate
from dbo.newdata$
group by postingdate;

SELECT count(*) as zero_variance,effectivedate
from dbo.newdata$
group by effectivedate;

select count(*) as zero_variance, customervendorcode
from dbo.newdata$
group by customervendorcode;

ALTER TABLE dbo.newdata$
DROP COLUMN customervendorcode;

select count(*) as zero_variance, Customervendorname
from dbo.newdata$
group by Customervendorname;

ALTER TABLE dbo.newdata$
DROP COLUMN Customervendorname;

select count(*) as zero_variance, LOB
from dbo.newdata$
group by LOB;

ALTER TABLE dbo.newdata$
DROP COLUMN LOB;

select count(*) as zero_variance, Region
from dbo.newdata$
group by Region;

ALTER TABLE dbo.newdata$
DROP COLUMN Region;

select count(*) as zero_variance, bptype
from dbo.newdata$
group by bptype;

ALTER TABLE dbo.newdata$
DROP COLUMN bptype;

select count(*) as zero_variance, City
from dbo.newdata$
group by City;

ALTER TABLE dbo.newdata$
DROP COLUMN city;

select count(*) as zero_variance, state
from dbo.newdata$
group by state;

ALTER TABLE dbo.newdata$
DROP COLUMN state;

select count(*) as zero_variance, FromWhscode
from dbo.newdata$ 
group by FromWhscode;

select count(*) as zero_variance, FromWhsName
from dbo.newdata$
group by FromWhsName;

select count(*) as zero_variance, TowhsCode
from dbo.newdata$
group by TowhsCode;

ALTER TABLE dbo.newdata$
DROP COLUMN Towhscode;

select count(*) as zero_variance, TOWhsName
from dbo.newdata$
group by TOWhsName;

ALTER TABLE dbo.newdata$
DROP COLUMN TOWhsName;

select count(*) as zero_variance, ModelTYPE
from dbo.newdata$
group by ModelTYPE;

select count(*) as zero_variance, TransferType
from dbo.newdata$
group by TransferType;

ALTER TABLE dbo.newdata$
DROP COLUMN TransferType;

select count(*) as zero_variance, U_Frt
from dbo.newdata$
group by U_Frt;

select count(*) as zero_variance, U_ActShipType
from dbo.newdata$
group by U_ActShipType;

select count(*) as zero_variance, PRODUCTCATEGORY
from dbo.newdata$
group by PRODUCTCATEGORY;

select count(*) as zero_variance, ItemCode
from dbo.newdata$
group by ItemCode;

select count(*) as zero_variance, Description
from dbo.newdata$
group by Description;

select count(*) as zero_variance, QUANTITY
from dbo.newdata$
group by QUANTITY;

select count(*) as zero_variance, UNIT
from dbo.newdata$
group by UNIT;

ALTER TABLE dbo.newdata$
DROP COLUMN UNIT;

select count(*) as zero_variance, RATE
from dbo.newdata$
group by RATE;

select count(*) as zero_variance, SOID
from dbo.newdata$
group by SOID;

select count(*) as zero_variance, SOCreationDate
from dbo.newdata$
group by SOCreationDate;

select count(*) as zero_variance, SODueDate
from dbo.newdata$
group by SODueDate;

select count(*) as zero_variance, U_DocStatus
from new_pallets
group by U_DocStatus;

select count(*) as zero_variance, U_SOTYPE
from dbo.newdata$
group by U_SOTYPE;

ALTER TABLE dbo.newdata$
DROP COLUMN U_SOTYPE;

select count(*) as zero_variance, BPCATEGORY
from dbo.newdata$
group by BPCATEGORY;

ALTER TABLE dbo.newdata$
DROP COLUMN BPCATEGORY;

select count(*) as zero_variance, DocumentType
from dbo.newdata$
group by DocumentType;

ALTER TABLE dbo.newdata$
DROP COLUMN DocumentType;

select count(*) as zero_variance, TRANSPORTERNAME
from dbo.newdata$
group by TRANSPORTERNAME;

select count(*) as zero_variance, U_GRNNO
from dbo.newdata$
group by U_GRNNO;

select count(*) as zero_variance, LoadingUnloading
from dbo.newdata$
group by LoadingUnloading;


select * from dbo.newdata$

select count(*) as zero_variance, U_AssetClass
from dbo.newdata$
group by U_AssetClass;

 ---EDA
---- Descriptive Statistics:
--- Mean
SELECT AVG(u_frt) AS average, MIN(u_frt) AS minimum, MAX(u_frt) AS maximum, COUNT(*) AS count FROM dbo.newdata$;
SELECT AVG(quantity) AS average, MIN(quantity) AS minimum, MAX(quantity) AS maximum, COUNT(*) AS count FROM dbo.newdata$;
SELECT AVG(rate) AS average, MIN(rate) AS minimum, MAX(rate) AS maximum, COUNT(*) AS count FROM dbo.newdata$;
SELECT AVG(loadingunloading) AS average, MIN(loadingunloading) AS minimum, MAX(loadingunloading) AS maximum, COUNT(*) AS count FROM dbo.newdata$;


---Data Distribution
---MODE
SELECT u_frt, COUNT(*) AS count FROM dbo.newdata$ GROUP BY u_frt

SELECT quantity, COUNT(*) AS count FROM dbo.newdata$ GROUP BY quantity
SELECT rate, COUNT(*) AS count FROM dbo.newdata$ GROUP BY rate
SELECT loadingunloading, COUNT(*) AS count FROM dbo.newdata$ GROUP BY loadingunloading
SELECT u_grnno, COUNT(*) AS count FROM dbo.newdata$ GROUP BY u_grnno

--- MEDIAN

SELECT
  PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY quantity) AS median_value
FROM
  dbo.newdata$;
  SELECT
  PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY rate) AS median_value
FROM
  dbo.newdata$;
  SELECT
  PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY loadingunloading) AS median_value
FROM
  dbo.newdata$;
  SELECT
  PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY u_grnno) AS median_value
FROM
  dbo.newdata$;

 ---Second Moment Business Decision
 ---variance

SELECT VAR(u_frt) AS variance FROM dbo.newdata$;
SELECT VAR(quantity) AS variance FROM dbo.newdata$;
SELECT VAR(rate) AS variance FROM dbo.newdata$;
SELECT VAR(u_grnno) AS variance FROM dbo.newdata$;

-- standard deviation
SELECT STDEV(u_frt) AS standard_deviation FROM dbo.newdata$;
SELECT STDEV(quantity) AS standard_deviation FROM dbo.newdata$;
SELECT STDEV(rate) AS standard_deviation FROM dbo.newdata$;
SELECT STDEV(u_grnno) AS standard_deviation FROM dbo.newdata$;



----Third moment business decision:

SELECT (1 / COUNT(*) * SUM(POWER(quantity - AVG(quantity), 3))) / POWER(STDEV(quantity), 3) AS skewness
FROM dbo.newdata$;


SELECT (1 / COUNT(*) * SUM(POWER(u_frt - AVG(u_frt), 3))) / POWER(STDEV(u_frt), 3) AS skewness
FROM dbo.newdata$;
SELECT (1 / COUNT(*) * SUM(POWER(rate - AVG(rate), 3))) / POWER(STDEV(rate), 3) AS skewness
FROM dbo.newdata$;
SELECT (1 / COUNT(*) * SUM(POWER(u_grnno - AVG(u_grnno), 3))) / POWER(STDEV(u_grnno), 3) AS skewness
FROM dbo.newdata$;


