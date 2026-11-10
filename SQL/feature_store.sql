WITH 
main_tb AS (
    SELECT *
    FROM workspace.default.{dataset}
),

standardize_variables AS (
    SELECT 
        customerID,

        -- IDENTITY
        -- gender, 
        (CASE WHEN lower(gender)='male' THEN 1 ELSE 0 END) AS isMale,
        SeniorCitizen,
        -- Partner,
        (CASE WHEN lower(Partner)='yes' THEN 1 ELSE 0 END) AS isPartner,
        -- Dependents,
        (CASE WHEN lower(Dependents)='yes' THEN 1 ELSE 0 END) AS HasDependents,
        MonthlyIncome,

        -- ACCOUNT
        -- tenure,
        TRY_CAST(tenure AS BIGINT) AS nbrTenure,
        -- Contract,
        (CASE 
            WHEN lower(Contract)='month-to-month' THEN 'M2M'
            WHEN lower(Contract)='one year' THEN 'one-year' 
            WHEN lower(Contract)='two year' THEN 'two-year' 
            ELSE NULL
        END) AS ContractTier,
        (CASE WHEN lower(Contract)='one year' OR lower(Contract)='two year' THEN 1 ELSE 0 END) AS isContractLong,
        (CASE WHEN lower(Contract)='one year' THEN 1 ELSE 0 END) AS isContractOneYear,
        (CASE WHEN lower(Contract)='two year' THEN 1 ELSE 0 END) AS isContractTwoYear,
        -- PaperlessBilling,
        (CASE WHEN lower(PaperlessBilling)='yes' THEN 1 ELSE 0 END) AS isPaperlessBilling,
        -- PaymentMethod,
        (CASE WHEN lower(PaymentMethod) LIKE '%automatic%' THEN 1 ELSE 0 END) AS isPaymentMethodAuto,
        (CASE WHEN lower(PaymentMethod) LIKE '%check%' THEN 1 ELSE 0 END) AS isPaymentMethodCheck,
        (CASE WHEN lower(PaymentMethod) LIKE 'bank transfer%' THEN 1 ELSE 0 END) AS isPaymentMethod_BankTransfer,
        (CASE WHEN lower(PaymentMethod) LIKE 'credit card%' THEN 1 ELSE 0 END) AS isPaymentMethod_CreditCard,
        (CASE WHEN lower(PaymentMethod) LIKE 'electronic check%' THEN 1 ELSE 0 END) AS isPaymentMethod_ElectronicCheck,
        (CASE WHEN lower(PaymentMethod) LIKE 'mailed check%' THEN 1 ELSE 0 END) AS isPaymentMethod_MailedCheck,

        -- PHONE SERVICES
        -- PhoneService,
        (CASE WHEN lower(PhoneService)='yes' THEN 1 ELSE 0 END) AS hasPhoneService,
        -- MultipleLines,
        (CASE 
            WHEN lower(MultipleLines)='yes' THEN 1 
            WHEN lower(MultipleLines)='no' THEN 0 
            WHEN lower(MultipleLines)='no phone service' THEN 0 
            ELSE NULL 
        END) AS hasMultipleLines,

        -- INTERNET SERVICES
        -- InternetService,
        (CASE WHEN lower(InternetService)='dsl' OR lower(InternetService)='fiber optic' THEN 1 ELSE 0 END) AS hasInternetService,
        (CASE WHEN lower(InternetService)='dsl' THEN 1 ELSE 0 END) AS isInternetService_DSL,
        (CASE WHEN lower(InternetService)='fiber optic' THEN 1 ELSE 0 END) AS isInternetService_Fiber,

        -- INTERNET SERVICES ADD-ON
        -- OnlineSecurity,
        (CASE WHEN lower(OnlineSecurity)='yes' THEN 1 ELSE 0 END) AS hasOnlineSecurity,
        -- OnlineBackup,
        (CASE WHEN lower(OnlineBackup)='yes' THEN 1 ELSE 0 END) AS hasOnlineBackup,
        -- DeviceProtection,
        (CASE WHEN lower(DeviceProtection)='yes' THEN 1 ELSE 0 END) AS hasDeviceProtection,
        -- TechSupport,
        (CASE WHEN lower(TechSupport)='yes' THEN 1 ELSE 0 END) AS hasTechSupport,
        -- StreamingTV,
        (CASE WHEN lower(StreamingTV)='yes' THEN 1 ELSE 0 END) AS hasStreamingTV,
        -- StreamingMovies,
        (CASE WHEN lower(StreamingMovies)='yes' THEN 1 ELSE 0 END) AS hasStreamingMovies,

        -- TARGET
        -- Churn,
        -- (CASE WHEN lower(Churn) = 'yes' THEN 1 ELSE 0 END) AS TargetChurn,

        -- FEEDBACK
        -- CustomerFeedback,
        -- TRIM(BOTH '"' FROM CustomerFeedback) AS CustomerFeedback_adj,

        -- BILLING
        MonthlyCharges,
        TotalCharges,
        (CASE 
            WHEN MonthlyCharges < 30 THEN '0-30'
            WHEN MonthlyCharges < 70 THEN '30-70'
            WHEN MonthlyCharges < 90 THEN '70-90'
            ELSE '90+'
        END) AS ChargeBand

    FROM main_tb
) ,

-- Add derived variables
derived_variables AS (
    SELECT 
        *,
        -- ACCOUNT
        (CASE WHEN nbrTenure <= 6 THEN 1 ELSE 0 END) AS isNewCustomer,
        (CASE 
            WHEN nbrTenure <=  6 THEN '1_0to6'
            WHEN nbrTenure <= 12 THEN '2_7to12'
            WHEN nbrTenure <= 24 THEN '3_13to24'
            WHEN nbrTenure <= 48 THEN '4_25to48'
            WHEN nbrTenure >  48 THEN '5_49plus'
            ELSE '6_other'
        END) AS tenureGroup,

        -- INTERNET SERVICES ADDON
        (hasOnlineSecurity + hasOnlineBackup + hasDeviceProtection + hasTechSupport + hasStreamingTV + hasStreamingMovies) AS nbrInternetServicesAddOnQty,
        (hasOnlineSecurity + hasOnlineBackup + hasDeviceProtection + hasTechSupport + hasStreamingTV + hasStreamingMovies) / 6.0 AS nbrInternetServicesAddOnQtyPct,
        (CASE WHEN hasOnlineSecurity + hasOnlineBackup + hasDeviceProtection = 3 THEN 1 ELSE 0 END) AS hasSecurityBundle,
        (CASE WHEN  hasStreamingTV + hasStreamingMovies > 1 THEN 1 ELSE 0 END) AS hasStreamingBundle,
        (hasStreamingTV + hasStreamingMovies) AS nbrStreamingCount,

        -- Service Tier
        (CASE 
            WHEN hasPhoneService = 1 AND hasInternetService = 1 THEN 'dual'
            WHEN hasPhoneService = 1 AND hasInternetService = 0 THEN 'phone_only'
            WHEN hasPhoneService = 0 AND hasInternetService = 1 THEN 'internet_only'
            ELSE NULL 
        END) AS ServiceTier,

        -- CROSS VARIABLES
        -- Total Spend
        (nbrTenure * MonthlyCharges) AS TotalSpend,
        TRY_DIVIDE(TotalCharges, TotalSpend) AS TotalSpendPct,

        -- Contract Tier
        (CASE WHEN ContractTier='M2M' AND (hasPhoneService = 1 AND hasInternetService = 1) THEN 1 ELSE 0 END) AS isM2M_dual,
        (CASE WHEN ContractTier='M2M' AND (hasPhoneService = 0 AND hasInternetService = 1) THEN 1 ELSE 0 END) AS isM2M_internetOnly,
        (CASE WHEN ContractTier='one-year' AND (hasPhoneService = 1 AND hasInternetService = 1) THEN 1 ELSE 0 END) AS isOneYear_dual,
        (CASE WHEN ContractTier='two-year' AND (hasPhoneService = 1 AND hasInternetService = 1) THEN 1 ELSE 0 END) AS isTwoYear_dual,

        -- Calcualted
        ROUND(TRY_DIVIDE(TotalCharges, MonthlyCharges),0) AS tenure_calc,
        TRY_DIVIDE(MonthlyCharges, MonthlyIncome) AS chargeRatio,
        (hasPhoneService + hasInternetService + hasMultipleLines + nbrInternetServicesAddOnQty) AS numServices,
        TRY_DIVIDE(MonthlyCharges, hasPhoneService + hasInternetService + hasMultipleLines + nbrInternetServicesAddOnQty) AS chargesPerService

    FROM standardize_variables
)

SELECT
    customerID,
    isMale,
    SeniorCitizen,
    isPartner,
    HasDependents,
    MonthlyIncome,
    (CASE WHEN nbrTenure IS NULL THEN tenure_calc ELSE nbrTenure END) AS nbrTenureMonths, 
    ContractTier,
    isContractLong,
    isContractOneYear,
    isContractTwoYear,
    isPaperlessBilling,
    isPaymentMethodAuto,
    isPaymentMethodCheck,
    isPaymentMethod_BankTransfer,
    isPaymentMethod_CreditCard,
    isPaymentMethod_ElectronicCheck,
    isPaymentMethod_MailedCheck,
    hasPhoneService,
    hasMultipleLines,
    hasInternetService,
    isInternetService_DSL,
    isInternetService_Fiber,
    hasOnlineSecurity,
    hasOnlineBackup,
    hasDeviceProtection,
    hasTechSupport,
    hasStreamingTV,
    hasStreamingMovies,
    MonthlyCharges,
    TotalCharges,
    isNewCustomer,
    tenureGroup,
    nbrInternetServicesAddOnQty,
    nbrInternetServicesAddOnQtyPct,
    hasSecurityBundle,
    hasStreamingBundle,
    nbrStreamingCount,
    ServiceTier,
    TotalSpend,
    TotalSpendPct,
    isM2M_dual,
    isM2M_internetOnly,
    isOneYear_dual,
    isTwoYear_dual,
    chargeRatio,
    numServices,
    chargesPerService
FROM derived_variables
