SET TRAN ISOLATION LEVEL READ UNCOMMITTED;
USE PayaAfzarPendarData
SELECT  DISTINCT
        us.UserName AS Sender,
        c.Ref AS Class,
        w.IsCopy,
        nt.Ref AS NodeType,
        w.LevelNo,
        w.IsPrivate,
        DATEPART(HOUR, w.ReceivedAt) AS Time,
        wgp.Ref AS SenderWorkgroup,
        psp.Ref AS SenderPost,
        DATEDIFF(MINUTE, p.ReceivedAt, w.ReceivedAt) AS WaitTime1,
        DATEDIFF(MINUTE, o.CreatedAt, w.ReceivedAt) AS WaitTime2,
        CASE
            WHEN EXISTS (
                            SELECT  *
                            FROM    dbo.AUT_Attachments
                            WHERE   ParentObjectId = w.ObjId
                                    AND CreatedAt <= w.ReceivedAt
                        ) THEN
                 1
            ELSE 0
        END AS HasAttachment,
        uo.UserName AS OwnerUser,
        wo.Ref AS OwnerWorkgroup,
        CASE WHEN p.ReceiverId = o.OwnerUser THEN 1 ELSE 0 END AS IsOwner,
        CASE WHEN LEN(o.BodyText) > 0 THEN 1 ELSE 0 END AS HasBody,
        CASE WHEN o.Parent IS NULL THEN 0 ELSE 1 END AS IsCopied,
        CASE WHEN o.DisplayName <> c.DisplayName THEN 1 ELSE 0 END AS HasCustomDisplayName,
        --CASE WHEN c.Ref = 'Task' THEN 1 ELSE 0 END,
        c.IsTask,
        CASE WHEN p.IsSent = 1 AND  p.SentAt < w.ReceivedAt THEN 1 ELSE 0 END AS IsSent,
        ur.Id AS ReceiverUser
FROM    (SELECT TOP (600000)   * FROM  dbo.AUT_WorkFlow ORDER BY   NodeId DESC) w
        INNER JOIN dbo.AUT_Objects o ON o.Id = w.ObjId
        INNER JOIN dbo.AUT_WorkFlow p ON w.ParentId = p.NodeId
        INNER JOIN dbo.AUT_Users ur ON ur.Id = w.ReceiverId
        INNER JOIN dbo.AUT_Users us ON us.Id = w.ReceiverId
        INNER JOIN dbo.AUT_Classes c ON c.Id = o.Class
        LEFT OUTER JOIN dbo.AUT_WorkFlowNodeTypes nt ON w.NodeTypeId = nt.Id
        LEFT OUTER JOIN dbo.AUT_WorkGroups wgp ON wgp.Id = p.ReceiverWorkGroupId
        LEFT OUTER JOIN dbo.AUT_Post psp ON psp.Id = p.ReceiverPostId
        INNER JOIN dbo.AUT_Users uo ON uo.Id = o.OwnerUser
        INNER JOIN dbo.AUT_WorkGroups wo ON wo.Id = o.OwnerWorkGroup
WHERE   o.BaseClass = 2
        AND w.ReceiverId <> p.ReceiverId;