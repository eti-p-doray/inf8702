#pragma once

#include <string>

#include <FreeImage.h>

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

  // Construit un objet image_file a partir du nom de fichier.
  image_file( const image_file& ) = delete;
  image_file( const std::string& name, Depth depth = k24Bits );
  image_file& operator=( const image_file& ) = delete;
  ~image_file();

  // Accede au tableau de donnees representant l'image.
  const void* data() const {
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
  size_t stride() const {
    return FreeImage_GetPitch( data_ );
  }

private:
  FIBITMAP* data_; // Objet image sous forme de BitMap FreeImage.
};

image_file::image_file( const std::string& name, Depth depth ) {
  // check the file signature and deduce its format
  // (the second argument is currently not used by FreeImage)
  FREE_IMAGE_FORMAT format = FreeImage_GetFileType( name.c_str(), 0 );

  if ( format == FIF_UNKNOWN ) {
    // no signature ?
    // try to guess the file format from the file extension
    format = FreeImage_GetFIFFromFilename( name.c_str() );
    if ( format == FIF_UNKNOWN ) {
      throw std::domain_error( "Format du fichier image \"" + name + "\" non supporte" );
    }
  }
  // ok, let's load the file
  FIBITMAP* dib = FreeImage_Load( format, name.c_str(), 0 );
  if ( dib == 0 ) {
    throw std::domain_error( "Erreur a la lecture du fichier \"" + name + "\"" );
  }

  switch (depth) {
    case k4Bits:
      data_ = FreeImage_ConvertTo4Bits( dib );
      break;
    case k8Bits:
      data_ = FreeImage_ConvertTo8Bits( dib );
      break;
    case kGreyScale:
      data_ = FreeImage_ConvertToGreyscale( dib );
      break;
    case k16Bits555:
      data_ = FreeImage_ConvertTo16Bits555( dib );
      break;
    case k16Bits565:
      data_ = FreeImage_ConvertTo16Bits565( dib );
      break;
    case k24Bits:
      data_ = FreeImage_ConvertTo24Bits( dib );
      break;
    case k32Bits:
      data_ = FreeImage_ConvertTo32Bits( dib );
      break;
    default:
      throw std::domain_error( "Invalid format" );
  }
  FreeImage_Unload( dib );
  if ( data_ == nullptr ) {
    throw std::domain_error( "Incapable de convertir le fichier \"" + name + "\"." );
  }
}

// Libere l'image.
image_file::~image_file() {
  FreeImage_Unload( data_ );
}
