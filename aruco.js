var AR = AR || (function() {
  var CV = this.CV || require('./cv').CV;

  var AR = {
    Marker: function(id, corners){
      this.id = id;
      this.corners = corners;
    },

    Detector: function(options){
      options = options || {};
      this.grey = new CV.Image();
      this.thres = new CV.Image();
      this.thresAlt = new CV.Image();
      this.thresGlobal = new CV.Image();
      this.blur = new CV.Image();
      this.homography = new CV.Image();
      this.binary = [];
      this.binaryAlt = [];
      this.binaryGlobal = [];
      this.contours = [];
      this.polys = [];
      this.candidates = [];
      this.dictionaryName = options.dictionaryName || 'ARUCO';
      this.dictionary = AR.Dictionary[this.dictionaryName];

      // Tunable detection parameters with more forgiving defaults for phone captures.
      this.params = {
        candidateMinSizeRatio: options.candidateMinSizeRatio || 0.12,
        epsilon: options.epsilon || 0.03,
        minEdgeLength: options.minEdgeLength || 14,
        nearDistance: options.nearDistance || 12,
        warpSize: options.warpSize || 70,
        borderWhiteMaxFraction: options.borderWhiteMaxFraction || 0.65,
        useMultiThreshold: (options.useMultiThreshold !== false),
        adaptiveBlockSize1: options.adaptiveBlockSize1 || 2,
        adaptiveOffset1: options.adaptiveOffset1 || 7,
        adaptiveBlockSize2: options.adaptiveBlockSize2 || 2,
        adaptiveOffset2: options.adaptiveOffset2 || 5,
        blurKernelSize: options.blurKernelSize || 3,
        useGlobalOtsuPass: (options.useGlobalOtsuPass !== false)
      };
    }
  };

  AR.Dictionary = {
    ARUCO: {
      size: 5,
      maxCorrectionBits: 0,
      bytesList: [
        [0x10,0x17,0x09,0x0e],[0x0a,0x0d,0x01,0x16],[0x13,0x09,0x0d,0x12],[0x0b,0x0e,0x06,0x11],
        [0x01,0x0c,0x15,0x14],[0x00,0x0f,0x04,0x1b],[0x08,0x0b,0x12,0x15],[0x0f,0x0a,0x03,0x1d],
        [0x0d,0x05,0x14,0x07],[0x02,0x14,0x15,0x09],[0x06,0x1d,0x02,0x05],[0x1c,0x01,0x13,0x0d],
        [0x19,0x06,0x07,0x0e],[0x1a,0x19,0x11,0x03],[0x17,0x1e,0x0b,0x08],[0x14,0x00,0x1e,0x1a]
      ]
    },
    ARUCO_MIP_36h12: {
      size: 6,
      maxCorrectionBits: 5,
      bytesList: [
        [48,6,226,186,70],[15,157,93,162,241],[49,89,220,108,164],[182,102,245,183,218],
        [198,188,87,114,219],[181,174,14,240,238],[49,9,16,230,79],[144,145,122,197,177],
        [73,175,60,66,199],[159,91,195,120,230],[79,194,98,194,80],[46,111,198,243,250],
        [187,114,169,54,28],[152,156,132,90,204],[12,31,211,79,233],[123,153,71,12,161]
      ]
    }
  };

  AR.Detector.prototype.detect = function(image){
    var markers = [], thresholdImages = [], thresholdBinaries = [], i;

    CV.grayscale(image, this.grey);

    // Pass 1: original adaptive threshold.
    CV.adaptiveThreshold(this.grey, this.thres,
      this.params.adaptiveBlockSize1,
      this.params.adaptiveOffset1);
    thresholdImages.push(this.thres);
    thresholdBinaries.push(this.binary);

    if (this.params.useMultiThreshold){
      // Pass 2: slight blur + softer adaptive threshold, helps under indoor noise.
      CV.gaussianBlur(this.grey, this.blur, this.blur, this.params.blurKernelSize);
      CV.adaptiveThreshold(this.blur, this.thresAlt,
        this.params.adaptiveBlockSize2,
        this.params.adaptiveOffset2);
      thresholdImages.push(this.thresAlt);
      thresholdBinaries.push(this.binaryAlt);

      // Pass 3: global Otsu on blurred grayscale, helps when adaptive breaks up borders.
      if (this.params.useGlobalOtsuPass){
        CV.threshold(this.blur, this.thresGlobal, CV.otsu(this.blur));
        thresholdImages.push(this.thresGlobal);
        thresholdBinaries.push(this.binaryGlobal);
      }
    }

    this.contours = [];
    this.polys = [];
    this.candidates = [];

    for (i = 0; i < thresholdImages.length; ++i){
      markers = markers.concat(this.detectOnThreshold(image, this.grey, thresholdImages[i], thresholdBinaries[i]));
    }

    return this.mergeMarkers(markers);
  };

  AR.Detector.prototype.detectOnThreshold = function(image, greyImage, thresholdImage, binaryStore){
    var contours, candidates;

    contours = CV.findContours(thresholdImage, binaryStore);
    candidates = this.findCandidates(
      contours,
      image.width * this.params.candidateMinSizeRatio,
      this.params.epsilon,
      this.params.minEdgeLength
    );
    candidates = this.clockwiseCorners(candidates);
    candidates = this.notTooNear(candidates, this.params.nearDistance);

    this.contours = this.contours.concat(contours);
    this.candidates = this.candidates.concat(candidates);

    return this.findMarkers(greyImage, candidates, this.params.warpSize);
  };

  AR.Detector.prototype.findCandidates = function(contours, minSize, epsilon, minLength){
    var candidates = [], len = contours.length, contour, poly, i;

    for (i = 0; i < len; ++ i){
      contour = contours[i];

      if (contour.length >= minSize){
        poly = CV.approxPolyDP(contour, contour.length * epsilon);
        this.polys.push(poly);

        if ((4 === poly.length) && (CV.isContourConvex ? CV.isContourConvex(poly) : true)){
          if (this.minEdgeLength(poly) >= minLength){
            candidates.push(this.refineCorners(poly));
          }
        }
      }
    }

    return candidates;
  };

  AR.Detector.prototype.refineCorners = function(poly){
    // Lightweight refinement: snap each corner to integer pixels and keep shape stable.
    // This is intentionally conservative so it does not distort the quad.
    var out = [], i;
    for (i = 0; i < poly.length; ++i){
      out.push({x: Math.round(poly[i].x), y: Math.round(poly[i].y)});
    }
    return out;
  };

  AR.Detector.prototype.clockwiseCorners = function(candidates){
    var len = candidates.length, dx1, dx2, dy1, dy2, swap, i;

    for (i = 0; i < len; ++ i){
      dx1 = candidates[i][1].x - candidates[i][0].x;
      dy1 = candidates[i][1].y - candidates[i][0].y;
      dx2 = candidates[i][2].x - candidates[i][0].x;
      dy2 = candidates[i][2].y - candidates[i][0].y;

      if ((dx1 * dy2 - dy1 * dx2) < 0){
        swap = candidates[i][1];
        candidates[i][1] = candidates[i][3];
        candidates[i][3] = swap;
      }
    }

    return candidates;
  };

  AR.Detector.prototype.notTooNear = function(candidates, minDist){
    var survivors = [], len = candidates.length, dist, dx, dy, i, j, k,
        areaI, areaJ;

    minDist *= minDist;

    for (i = 0; i < len; ++ i){
      candidates[i].rejected = false;
    }

    for (i = 0; i < len; ++ i){
      if (candidates[i].rejected) continue;

      for (j = i + 1; j < len; ++ j){
        if (candidates[j].rejected) continue;

        dist = 0;
        for (k = 0; k < 4; ++ k){
          dx = candidates[i][k].x - candidates[j][k].x;
          dy = candidates[i][k].y - candidates[j][k].y;
          dist += dx * dx + dy * dy;
        }

        if ((dist / 4) < minDist){
          areaI = this.quadArea(candidates[i]);
          areaJ = this.quadArea(candidates[j]);
          if (areaI >= areaJ){
            candidates[j].rejected = true;
          } else {
            candidates[i].rejected = true;
            break;
          }
        }
      }
    }

    for (i = 0; i < len; ++ i){
      if (!candidates[i].rejected){
        survivors.push(candidates[i]);
      }
    }

    return survivors;
  };

  AR.Detector.prototype.findMarkers = function(imageSrc, candidates, warpSize){
    var markers = [], len = candidates.length, candidate, marker, i;

    for (i = 0; i < len; ++ i){
      candidate = candidates[i];
      if (CV.warp){
        CV.warp(imageSrc, this.homography, candidate, warpSize);
      }
      CV.threshold(this.homography, this.homography, CV.otsu(this.homography));

      marker = this.getMarker(this.homography, candidate);
      if (marker){
        markers.push(marker);
      }
    }

    return markers;
  };

  AR.Detector.prototype.getMarker = function(imageSrc, candidate){
    var width = (imageSrc.width / 7) >>> 0,
        minZero = Math.floor(width * width * this.params.borderWhiteMaxFraction),
        bits = [], rotations = [], distances = [],
        square, pair, inc, i, j;

    for (i = 0; i < 7; ++ i){
      inc = (i === 0 || i === 6) ? 1 : 6;

      for (j = 0; j < 7; j += inc){
        square = {x: j * width, y: i * width, width: width, height: width};
        if (CV.countNonZero && CV.countNonZero(imageSrc, square) > minZero){
          return null;
        }
      }
    }

    for (i = 0; i < 5; ++ i){
      bits[i] = [];
      for (j = 0; j < 5; ++ j){
        square = {x: (j + 1) * width, y: (i + 1) * width, width: width, height: width};
        bits[i][j] = (CV.countNonZero && CV.countNonZero(imageSrc, square) > ((width * width) >> 1)) ? 1 : 0;
      }
    }

    rotations[0] = bits;
    distances[0] = this.hammingDistance(rotations[0]);
    pair = {first: distances[0], second: 0};

    for (i = 1; i < 4; ++ i){
      rotations[i] = this.rotate(rotations[i - 1]);
      distances[i] = this.hammingDistance(rotations[i]);
      if (distances[i] < pair.first){
        pair.first = distances[i];
        pair.second = i;
      }
    }

    if (pair.first > (this.dictionary.maxCorrectionBits || 0)){
      return null;
    }

    return new AR.Marker(
      this.mat2id(rotations[pair.second]),
      this.rotate2(candidate, 4 - pair.second)
    );
  };

  AR.Detector.prototype.mergeMarkers = function(markers){
    var bestById = {}, i, m, key;

    for (i = 0; i < markers.length; ++i){
      m = markers[i];
      key = String(m.id);
      if (!bestById[key] || this.quadArea(m.corners) > this.quadArea(bestById[key].corners)){
        bestById[key] = m;
      }
    }

    markers = [];
    for (key in bestById){
      if (bestById.hasOwnProperty(key)){
        markers.push(bestById[key]);
      }
    }
    return markers;
  };

  AR.Detector.prototype.hammingDistance = function(bits){
    var ids = [[1,0,0,0,0],[1,0,1,1,1],[0,1,0,0,1],[0,1,1,1,0]],
        dist = 0, sum, minSum, i, j, k;

    for (i = 0; i < 5; ++ i){
      minSum = Infinity;

      for (j = 0; j < 4; ++ j){
        sum = 0;
        for (k = 0; k < 5; ++ k){
          sum += bits[i][k] === ids[j][k] ? 0 : 1;
        }
        if (sum < minSum){
          minSum = sum;
        }
      }

      dist += minSum;
    }

    return dist;
  };

  AR.Detector.prototype.mat2id = function(bits){
    var id = 0, i;
    for (i = 0; i < 5; ++ i){
      id <<= 1;
      id |= bits[i][1];
      id <<= 1;
      id |= bits[i][3];
    }
    return id;
  };

  AR.Detector.prototype.rotate = function(src){
    var dst = [], len = src.length, i, j;
    for (i = 0; i < len; ++ i){
      dst[i] = [];
      for (j = 0; j < src[i].length; ++ j){
        dst[i][j] = src[src[i].length - j - 1][i];
      }
    }
    return dst;
  };

  AR.Detector.prototype.rotate2 = function(src, rotation){
    var dst = [], len = src.length, i;
    for (i = 0; i < len; ++ i){
      dst[i] = src[(rotation + i) % len];
    }
    return dst;
  };

  AR.Detector.prototype.minEdgeLength = function(poly){
    var len = poly.length, i = 0, j = len - 1,
        min = Infinity, d, dx, dy;

    for (; i < len; j = i ++){
      dx = poly[i].x - poly[j].x;
      dy = poly[i].y - poly[j].y;
      d = dx * dx + dy * dy;
      if (d < min){
        min = d;
      }
    }

    return Math.sqrt(min);
  };

  AR.Detector.prototype.quadArea = function(quad){
    var area = 0, i, j;
    for (i = 0, j = quad.length - 1; i < quad.length; j = i++){
      area += (quad[j].x + quad[i].x) * (quad[j].y - quad[i].y);
    }
    return Math.abs(area * 0.5);
  };

  return AR;
}());
