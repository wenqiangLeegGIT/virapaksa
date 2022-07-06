
# evalscript for S2 nir/red/greed/blue band, data type is UINT16 ,data format is tiff
es_s2_4bands = """
    //VERSION=3
    function setup(){
        return {
            input: [
            {   
                bands:["B08","B04", "B03", "B02","SCL"],
                units: ["REFLECTANCE","REFLECTANCE","REFLECTANCE","REFLECTANCE","DN"] // default units
            }
            ],
            output: [
                {
                id: "MultiSpectral",
                bands: 4,
                sampleType: "FLOAT32"
                },
                {
                id: "SCL",
                bands: 1,
                sampleType: "UINT16"
                },
                ]
            }
    }
                
    function evaluatePixel(sample) {
        // Multiply input reflectance values  by 65535 to return the band values clamped to [0 ,65535] unsigned 16 bit range. 
        return {
            MultiSpectral: [sample.B02, sample.B03,  sample.B04,  sample.B08],
            SCL: [sample.SCL],
        }
    }
"""

# evalscript for download  nir/red/greed/blue band of S2/Landsat8-9/MCD43A4 of a certain area, data type is UINT16 ,data format is tiff
es_slm = """
    //VERSION=3
    function setup(){
        return {
            input: [
                {   
                    datasource: "l2a",
                    bands:["B08","B04", "B03", "B02"],
                    units: "REFLECTANCE"
                },
                {   
                    datasource: "ls8",
                    bands:["B05","B04", "B03", "B02"],
                    units: "REFLECTANCE"
                },
                {   
                    datasource: "modis",
                    bands:["B01","B02", "B03", "B04"],
                    units: "REFLECTANCE"
                }
            ],
            output: [
                {
                    id: "landsat8_9_c2_l2",
                    bands: 4,
                    sampleType: "FLOAT32"
                },
                {
                    id: "s2_l2a",
                    bands: 4,
                    sampleType: "FLOAT32"
                },
                {
                    id: "modis_mcd43a4_006",
                    bands: 4,
                    sampleType: "FLOAT32"
                },
                ]
            }
    }
                
    function evaluatePixel(sample) {
        var ls = sample.ls8[0]
        var s2 = sample.l2a[0]
        var ms = sample.modis[0]
        return {
            landsat8_9_c2_l2: [ls.B02, ls.B03,  ls.B04, ls.B05] ,
            s2_l2a: [s2.B02, s2.B03,  s2.B04,  s2.B08],
            modis_mcd43a4_006: [ms.B01, ms.B02,  ms.B03,  ms.B04]
        }
    }

"""

es_zp = """
    //VERSION=3
    function setup(){
        return {
            input: [
                {   
                    bands:["B01","B02", "B03", "B04"],
                    units: "REFLECTANCE"
                }
            ],
            output: [
                {
                    id: "modis_mcd43a4_006",
                    bands: 4,
                    sampleType: "FLOAT32"
                },
                ]
            }
    }

    function evaluatePixel(sample) {
        return  {modis_mcd43a4_006:[sample.B01, sample.B02,  sample.B03,  sample.B04]}
    }

"""