USE [Modern_Master]
GO
/****** Object:  StoredProcedure [bi].[USP_Report_AssociationRules]    Script Date: 5/29/2022 3:03:15 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
-- =============================================
-- Author:		Name
-- Create date: 
-- Description:	
-- =============================================
ALTER PROCEDURE [bi].[USP_Report_AssociationRules]
    @IdBrList BIGINT = NULL,
    @IdPrdToPlcList NVARCHAR(MAX) = NULL,
    @CurrentIdPrs BIGINT = NULL,
    @CurrentIdPlc BIGINT = NULL,
    @CurrentIdPrd BIGINT = NULL,
    @CurrentIdPrdToPlc BIGINT = NULL,
    @CurrentIdUser BIGINT = NULL,
    @CurrentIdBr BIGINT = NULL
AS
BEGIN
    -- SET NOCOUNT ON added to prevent extra result sets from
    -- interfering with SELECT statements.
    SET NOCOUNT ON;

    EXEC dbo.USP_checkUserAccessForReports @CurrentIdPrs = @CurrentIdPrs,
                                           @CurrentIdPlc = @CurrentIdPlc,
                                           @CurrentIdPrd = @CurrentIdPrd,
                                           @CurrentIdPrdToPlc = @CurrentIdPrdToPlc,
                                           @CurrentIdUser = @CurrentIdUser,
                                           @CurrentIdBr = @CurrentIdBr,
                                           @IdPrdToPlcList = @IdPrdToPlcList OUTPUT,
                                           @IdBrList = @IdBrList OUTPUT;


    SELECT ps.CuPrs,
           ps.AbrPrs,
           ps.DscPrs,
           c.ClusterIndex
    FROM bi.PrsClusters c
        INNER JOIN dbo.PrsSpc ps
            ON ps.IdPrs = c.IdPrs
    WHERE ps.IdPlc = @CurrentIdPlc
	ORDER BY c.ClusterIndex, ps.CuPrs;

END

