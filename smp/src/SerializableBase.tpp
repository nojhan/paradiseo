/*
<SerializableBase.tpp>
Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2012

Eremey Valetov

This software is governed by the CeCILL license under French law and
abiding by the rules of distribution of free software.  You can  use,
modify and/ or redistribute the software under the terms of the CeCILL
license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info".

ParadisEO WebSite : http://paradiseo.gforge.inria.fr
Contact: paradiseo-help@lists.gforge.inria.fr
*/

template<class T>
SerializableBase<T>::SerializableBase() : _value() {}

template<class T>
SerializableBase<T>::SerializableBase(T base) : _value(std::move(base)) {}

template<class T>
SerializableBase<T>::~SerializableBase() = default;

template<class T>
SerializableBase<T>::operator T&() {
    return _value;
}

template<class T>
void SerializableBase<T>::setValue(const T& newValue) {
    _value = newValue;
}

template<class T>
void SerializableBase<T>::unpack(const eoserial::Object* obj) {
    eoserial::unpack(*obj, "value", _value);
}

template<class T>
eoserial::Object* SerializableBase<T>::pack() const {
    eoserial::Object* obj = new eoserial::Object;
    obj->add("value", eoserial::make(_value));
    return obj;
}
