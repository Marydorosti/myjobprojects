SET TRAN ISOLATION LEVEL READ UNCOMMITTED;
USE Modern_Master

SELECT  h.IdIvcHdr, g.IdGds, d.EffectiveConfirmedAmount AS Amount
FROM    dbo.SlsIvcDtl d
        INNER JOIN dbo.SlsIvcHdr h ON h.IdIvcHdr = d.IdIvcHdr
        INNER JOIN GdsSpc g ON g.IdGds = d.IdGds ORDER BY h.IdIvcHdr;



SELECT * FROM dbo.[bi.GdsSpcBasket]
ALTER TABLE dbo.[bi.GdsSpcBasket]

ADD IdGds2 NVARCHAR(255)


