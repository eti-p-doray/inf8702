#pragma once

#include <utility>
#include <type_traits>

#include "acier/type_traits.hpp"

namespace acier {

template <class T, class = when<true>>
class compressed_member {
 public:
  template <class... Args, when< std::is_constructible<T, Args...>{} > = 0>
  constexpr compressed_member(Args&&... args)
      : member(std::forward<Args>(args)...) {}

 	constexpr 			T& get()
 	{ return member; }
 	constexpr const T& get() const
 	{ return member; }

 	explicit operator 			T&()
 	{ return member; }
 	explicit operator const T&() const
 	{ return member; }

 private:
 	T member;
};

template <class T>
class compressed_member<T, when< std::is_empty<T>{} && !std::is_final<T>{} >> 
		: private T {
 public:

  template <class... Args, when< std::is_constructible<T, Args...>{} > = 0>
  constexpr compressed_member(Args&&... args)
      : T(std::forward<Args>(args)...) {}

	constexpr 			T& get()
 	{ return *this; }
 	constexpr const T& get() const
 	{ return *this; }

 	explicit operator 			T&()
 	{ return *this; }
 	explicit operator const T&() const
 	{ return *this; }
 	
};

}
