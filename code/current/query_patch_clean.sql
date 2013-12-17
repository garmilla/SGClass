-- This query does a table JOIN between the imaging (PhotoObj) and spectra
-- (SpecObj) tables and includes the necessary columns in the SELECT to upload
-- the results to the DAS (Data Archive Server) for FITS file retrieval.
SELECT
   p.objid,p.type,p.ra,p.dec,p.u,p.g,p.r,p.i,p.z,
   p.err_u,p.err_g,p.err_r,p.err_i,p.err_z
FROM PhotoObj as p
WHERE 
   p.r BETWEEN 20.0 AND 21.0
   AND p.mode = 1
   AND p.ra BETWEEN 348.0 AND 350.0
   AND p.dec BETWEEN 0.0 AND 2.0
   AND ((p.flags & 0x10000000) != 0)
   AND (((p.flags & 0x400000000000) = 0) OR (p.psfmagerr_g <= 0.2))
   AND (((p.flags & 0x100000000000) = 0) OR (p.flags & 0x1000) = 0)
