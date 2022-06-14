--SELECT * FROM dbo.vw_PrdToPlc

--DECLARE @Dt_Start DATE = dbo.dfn_ConvertFromPersianDate(1398, 1, 1);
--DECLARE @Dt_End DATE = dbo.dfn_ConvertFromPersianDate(1398, 12, 29);
--DECLARE @IdPrdToPlc BIGINT = (
--                                 SELECT IdPrdToPlc
--                                 FROM   dbo.vw_PrdToPlc
--                                 WHERE  IdPlc = 16801
--                                        AND @Dt_Start BETWEEN Dt_Prd_From_M AND Dt_Prd_To_M
--                             );

SELECT      x.IdPrsClient,
            --ps.CuPrs AS CustomerCode,
            x.Dt_Effect,
            x.CountIvc,
            x.CountGds,
            x.CountClassGds,
            x.SumAmount,
            x.SumPrice,
            c.CuClass,
            rg.CuItem AS DscReg
FROM        (
                SELECT      h.IdPrsClient,
                            CONVERT(DATE, h.Dt_Effect) AS Dt_Effect,
                            COUNT(*) AS CountIvc,
                            SUM(g.C) AS CountGds,
                            SUM(g.CC) AS CountClassGds,
                            SUM(g.A) AS SumAmount,
                            SUM(hs.SumTotalWithoutTax) AS SumPrice,
                            SUM(hs.SumDiscountTotal) AS SumDiscount
                FROM        dbo.SlsIvcHdr h
                            INNER JOIN dbo.SlsIvcHdrSumView hs ON hs.IdIvcHdr = h.IdIvcHdr
                            CROSS APPLY (
                                            SELECT  COUNT(DISTINCT IdGds) AS C,
                                                    COUNT(d.EffectiveConfirmedAmount) AS A,
                                                    COUNT(DISTINCT gc.IdClass) AS CC
                                            FROM    dbo.SlsIvcDtl d
                                                    CROSS APPLY (
                                                                    SELECT  TOP (1)
                                                                            ic.IdClass
                                                                    FROM    dbo.ItemToClass ic
                                                                    WHERE   ic.IdItem = d.IdGds
                                                                            AND ic.Type = 2
                                                                ) gc
                                            WHERE   d.IdIvcHdr = h.IdIvcHdr
                                                    AND d.IdGds IS NOT NULL
                                        ) g
                --OUTER APPLY (
                --                SELECT  pv.IdPrs
                --                FROM    dbo.SlsIvcVisitor iv
                --                        INNER JOIN dbo.SlsVisitorRuleHdr vh ON vh.IdVisitorRuleHdr = iv.IdVisitorRuleHdr
                --                        INNER JOIN dbo.PrsSpc pv ON pv.IdPrs = vh.IdPrs
                --                WHERE   iv.IdIvcHdr = h.IdIvcHdr
                --            ) vs
                --WHERE       h.Dt_Effect BETWEEN @Dt_Start AND @Dt_End
                GROUP BY    h.IdPrsClient,
                            CONVERT(DATE, h.Dt_Effect)
            ) x
            INNER JOIN dbo.PrsSpc ps ON ps.IdPrs = x.IdPrsClient
            CROSS APPLY (
                            SELECT  TOP 1
                                    c.CuClass
                            FROM    dbo.ItemToClass ic
                                    INNER JOIN dbo.Class c ON c.IdClass = ic.IdClass
                                                              AND   ic.IdItem = ps.IdPrs
                                                              AND   ic.Type = 1
                                                              AND   c.ClsType = 1
                        ) c
            CROSS APPLY (
                            SELECT  TOP 1
                                    r.CuItem
                            FROM    dbo.PrsToItemRegion pir
                                    INNER JOIN dbo.Item r ON r.IdItem = pir.IdItemRegion
                            WHERE   pir.IdPrs = x.IdPrsClient
                                    AND r.Type = 6
                        ) rg
ORDER BY    x.Dt_Effect,
            ps.IdPrs;