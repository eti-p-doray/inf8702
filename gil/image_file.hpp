#pragma once

#include <string>

#include <FreeImage.h>

#include "gil/mat.hpp"
#include "gil/vec.hpp"

namespace gil {

/**
* Cette classe permet d'ouvrir un fichier image et d'obtenir un tableau (bitmap)
*  grace a la librairie FreeImage.
*  Le tableau d'image est sous un format RGA ou RGBA row-wise
**/
class image_file {
public:
  enum Depth {
    k4Bits,
    k8Bits,
    kGreyScale,
    k16Bits555,
    k16Bits565,
    k24Bits,
    k32Bits,
  };
  enum Format {
    kUnknown = FIF_UNKNOWN,
    kBmp = FIF_BMP,
    kCut = FIF_CUT,
    kDos = FIF_DDS,
    kExr = FIF_EXR,
    kFaxG3 = FIF_FAXG3,
    kGif = FIF_GIF,
    kHdr = FIF_HDR,
    kIco = FIF_ICO,
    kIff = FIF_IFF,
    kJ2K = FIF_J2K,
    kJng = FIF_JNG,
    kJp2 = FIF_JP2,
    kJpeg = FIF_JPEG,
    kKoala = FIF_KOALA,
    kMng = FIF_MNG,
    kPbm = FIF_PBM,
    kPbmRaw = FIF_PBMRAW,
    kPcd = FIF_PCD,
    kPcx = FIF_PCX,
    kPfm = FIF_PFM,
    kPgm = FIF_PGM,
    kPgmRaw = FIF_PGMRAW,
    kPict = FIF_PICT,
    KPng = FIF_PNG,
  };

  // Construit un objet image_file a partir du nom de fichier.
  image_file() = default;
  image_file(const std::string& filename);
  image_file(gil::vec2<size_t> size, size_t depth,
             unsigned red_mask = 0,
             unsigned green_mask = 0,
             unsigned blue_mask = 0) {
    data_ = FreeImage_Allocate(int(size[1]), int(size[0]), int(depth),
      red_mask, green_mask, blue_mask);
    if ( data_ == nullptr )
      throw std::domain_error("Allocation failed");
  }
  image_file(gil::vec2<size_t> size, size_t pitch, const uint8_t* data);
  template <class T>
  image_file(mat_cview<T> that,
             unsigned red_mask = 0,
             unsigned green_mask = 0,
             unsigned blue_mask = 0) {
    data_ = FreeImage_Allocate(int(that.cols()), int(that.rows()), sizeof(T),
      red_mask, green_mask, blue_mask);
    uint8_t* d_first = data();
    for (size_t i = 0; i < that.rows(); ++i) {
      std::copy_n(that.row_cbegin(i), that.cols(), reinterpret_cast<gil::vec3b*>(d_first));
      d_first += pitch();
    }
  }
  image_file(mat_cview<gil::vec4b> that) {
    data_ = FreeImage_AllocateT(FIT_BITMAP, int(that.cols()), int(that.rows()), 32, 0xff0000, 0x00ff00, 0x0000ff);
    if ( data_ == nullptr )
      throw std::domain_error("Allocation failed");
    uint8_t* d_first = data();
    for (size_t i = 0; i < that.rows(); ++i) {
      std::copy_n(that.row_cbegin(i), that.cols(), reinterpret_cast<gil::vec4b*>(d_first));
      d_first += pitch();
    }
  }
  ~image_file();
  
  image_file(const image_file&) = delete;
  image_file(image_file&&);
  
  image_file& operator=(const image_file&) = delete;
  image_file& operator=(image_file&&);
  
  template <class T, class Alloc>
  operator mat<T, Alloc>() {
    assert(depth() == 8 * sizeof(T));
    return {{height(), width()}, pitch(), reinterpret_cast<const uint8_t*>(data())};
  }
  
  image_file convert(Depth depth = k24Bits) const;
  
  bool save(const std::string& filename, Format format, int flags = 0) const;

  // Accede au tableau de donnees representant l'image.
  const uint8_t* data() const {
    return FreeImage_GetBits( data_ );
  }
  uint8_t* data() {
    return FreeImage_GetBits( data_ );
  }

  // Accede a la largeur (en pixel) de l'image.
  size_t width() const {
    return FreeImage_GetWidth( data_ );
  }
  // Accede a la hauteur (en pixel) de l'image.
  size_t height() const {
    return FreeImage_GetHeight( data_ );
  }
  // Accede au nombre d'element (3 pou RBG ou 4 pour RGBA) dans chaque pixel.
  size_t depth() const {
    return FreeImage_GetBPP( data_ );
  }
  // Accede a l'ecart (en nombre de valeur) entre chaque range de donnees.
  size_t pitch() const {
    return FreeImage_GetPitch( data_ );
  }

private:
  FIBITMAP* data_ = nullptr; // Objet image sous forme de BitMap FreeImage.
};

image_file::image_file(const std::string& name) {
  // check the file signature and deduce its format
  // (the second argument is currently not used by FreeImage)
  FREE_IMAGE_FORMAT format = FreeImage_GetFileType( name.c_str(), 0 );

  if ( format == FIF_UNKNOWN ) {
    // no signature ?
    // try to guess the file format from the file extension
    format = FreeImage_GetFIFFromFilename( name.c_str() );
    if ( format == FIF_UNKNOWN ) {
      throw std::domain_error( "Unsupported file format" );
    }
  }
  // ok, let's load the file
  data_ = FreeImage_Load( format, name.c_str(), 0 );
  if ( data_ == 0 ) {
    throw std::domain_error( "Error reading file \"" + name + "\"" );
  }
}

image_file::image_file(image_file&& that) {
  std::swap(data_, that.data_);
}

image_file image_file::convert(Depth depth) const {
  image_file that;
  switch (depth) {
    case k4Bits:
      that.data_ = FreeImage_ConvertTo4Bits( data_ );
      break;
    case k8Bits:
      that.data_ = FreeImage_ConvertTo8Bits( data_ );
      break;
    case kGreyScale:
      that.data_ = FreeImage_ConvertToGreyscale( data_ );
      break;
    case k16Bits555:
      that.data_ = FreeImage_ConvertTo16Bits555( data_ );
      break;
    case k16Bits565:
      that.data_ = FreeImage_ConvertTo16Bits565( data_ );
      break;
    case k24Bits:
      that.data_ = FreeImage_ConvertTo24Bits( data_ );
      break;
    case k32Bits:
      that.data_ = FreeImage_ConvertTo32Bits( data_ );
      break;
    default:
      throw std::domain_error( "Invalid format" );
  }
  return that;
}

bool image_file::save(const std::string& filename, Format format, int flags) const {
  return FreeImage_Save(FREE_IMAGE_FORMAT(format), data_, filename.c_str(), flags);
}

// Libere l'image.
image_file::~image_file() {
  FreeImage_Unload( data_ );
}


}
